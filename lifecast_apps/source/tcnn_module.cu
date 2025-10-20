// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "tcnn_module.h"

#include <c10/cuda/CUDAGuard.h>
#include <string>

#include "logger.h"
#include "util_torch.h"

namespace p11 { namespace nerf {

// We are initializing to 0; this is unused
//static constexpr int kTCNNSeed = 1337;  // This is the default used by tinycudann's python module

class ContextWrapper : public torch::CustomClassHolder {
 public:
  ContextWrapper(tcnn::cpp::Context&& context) : context(std::move(context)) {}
  tcnn::cpp::Context& get() { return context; }
  const tcnn::cpp::Context& get() const { return context; }

 private:
  tcnn::cpp::Context context;
};

TORCH_LIBRARY(tcnn, m) { m.class_<ContextWrapper>("ContextWrapper"); }

// The "BackwardFunction" allows for 2nd-order gradients in some scenarios. So we only use the
// `forward` portion during the normal backward pass. This structure was preserved during the port
// but we don't really need it. It could be simplified. see
// https://github.com/NVlabs/tiny-cuda-nn/issues/58
class TcnnBackwardFunction : public torch::autograd::Function<TcnnBackwardFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::AutogradContext* ctx_fwd,
      torch::Tensor doutput,
      torch::Tensor input,
      torch::Tensor params,
      torch::Tensor output)
  {
    auto loss_scale = ctx_fwd->saved_data["loss_scale"].toDouble();

    ctx->saved_data["loss_scale"] = loss_scale;
    ctx->saved_data["native_ctx"] = ctx_fwd->saved_data["native_ctx"];
    ctx->saved_data["native_tcnn_module"] = ctx_fwd->saved_data["native_tcnn_module"];

    ctx->save_for_backward({input, params, doutput});

    torch::AutoGradMode enable_grad(false);

    auto scaled_grad = doutput * loss_scale;

    c10::Device device = input.device();
    const c10::cuda::CUDAGuard device_guard{device};
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    unsigned int batch_size = input.size(0);
    torch::Tensor input_grad{};

    // NOTE: for some reason libtorch throws an error if input_grad is undefined, even if
    // input.requires_grad() is false

    //if (input.requires_grad()) {
      input_grad = torch::zeros(
          {batch_size, input.size(1)},
          torch::TensorOptions().dtype(torch::kFloat32).device(device));
    //}

    auto module =
        reinterpret_cast<tcnn::cpp::Module*>(ctx_fwd->saved_data["native_tcnn_module"].toInt());

    torch::Tensor params_grad;
    if (params.requires_grad()) {
      params_grad = torch::zeros(
          {static_cast<long long>(module->n_params())}, // TODO: params.size(1) ?
          torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }

    auto ctx_wrapper = ctx->saved_data["native_ctx"].toCustomClass<ContextWrapper>();

    if (input.requires_grad() || params.requires_grad()) {
      module->backward(
          stream,
          ctx_wrapper->get(),
          batch_size,
          input.requires_grad() ? input_grad.data_ptr<float>() : nullptr,
          doutput.data_ptr<float>(),
          params.requires_grad() ? params_grad.data_ptr<float>() : nullptr,
          input.data_ptr<float>(),
          output.data_ptr<float>(),
          params.data_ptr<float>());
    }

    input_grad = input_grad.defined() ? input_grad / loss_scale : input_grad;
    params_grad = params_grad.defined() ? params_grad / loss_scale : params_grad;

    return {input_grad, params_grad};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output)
  {
    XCHECK(false) << "This shouldn't be running; we don't do a backward backward pass";
    return {};
  }
};

class TcnnFunction : public torch::autograd::Function<TcnnFunction> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      tcnn::cpp::Module* module,
      torch::Tensor input,
      torch::Tensor params,
      float loss_scale)
  {
    ctx->set_materialize_grads(false);

    c10::Device device = input.device();
    const c10::cuda::CUDAGuard device_guard{device};
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    unsigned int batch_size = input.size(0);

    torch::Tensor output =
        torch::zeros({batch_size, module->n_output_dims()}, torch::kFloat32).to(device);

    tcnn::cpp::Context native_ctx;

    const bool requires_grad = input.requires_grad() || params.requires_grad();
    if (!requires_grad) {
      module->inference(
          stream,
          batch_size,
          input.data_ptr<float>(),
          output.data_ptr<float>(),
          params.data_ptr<float>());
    } else {
      native_ctx = module->forward(
          stream,
          batch_size,
          input.data_ptr<float>(),
          output.data_ptr<float>(),
          params.data_ptr<float>(),
          requires_grad);
    }

    ctx->save_for_backward({input, params, output});
    ctx->saved_data["native_tcnn_module"] = reinterpret_cast<int64_t>(module);

    auto ctx_wrapper = c10::make_intrusive<ContextWrapper>(std::move(native_ctx));
    ctx->saved_data["native_ctx"] = ctx_wrapper;
    ctx->saved_data["loss_scale"] = loss_scale;

    return output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output)
  {
    auto doutput = grad_output[0];
    if (!doutput.is_cuda()) {
      //XPLWARN << "grad_output should be on CUDA but wasn't";
      doutput = doutput.cuda();
    }

    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto params = saved[1];
    auto output = saved[2];

    XCHECK(doutput.defined());
    XCHECK(input.defined());
    XCHECK(params.defined());
    XCHECK(output.defined());

    // I'm not sure why there's this extra layer of indirection
    auto results = TcnnBackwardFunction::apply(ctx, doutput, input, params, output);

    return {torch::Tensor{}, results[0], results[1], torch::Tensor{}};
  }
};

TcnnModule::TcnnModule(
    const torch::DeviceType device, tcnn::cpp::Module* module, const char* params_name)
    : module(module),
      params(torch::zeros(
          {static_cast<long long>(module->n_params())},
          torch::TensorOptions().dtype(torch::kFloat32).device(device))),
      loss_scale(tcnn::cpp::default_loss_scale(tcnn::cpp::Precision::Fp32))
{
  // Use zero-initialized params instead of TCNN's
  //module->initialize_params(kTCNNSeed, params.data_ptr<float>());
  register_parameter(params_name, params, true);
}

torch::Tensor TcnnModule::forward(torch::Tensor input)
{
  int batch_size = input.size(0);
  int padded_batch_size = (batch_size + tcnn::BATCH_SIZE_GRANULARITY - 1) /
                          tcnn::BATCH_SIZE_GRANULARITY *
                          tcnn::BATCH_SIZE_GRANULARITY;  // TODO Check this
  
  torch::Tensor input_padded;
  if (batch_size == padded_batch_size) {
    input_padded = input;
  } else {
    input_padded = torch::nn::functional::pad(
        input,
        torch::nn::functional::PadFuncOptions({0, 0, 0, padded_batch_size - batch_size})
            .mode(torch::kConstant));
  }

  //XCHECK(!at::isnan(input).any().item<bool>());

  torch::Tensor output = TcnnFunction::apply(
      module.get(),
      input_padded.to(torch::kFloat32).contiguous(),
      params.to(torch::kFloat32).contiguous(),
      loss_scale);

  auto sliced_output = output.index({torch::indexing::Slice(torch::indexing::None, batch_size), torch::indexing::Ellipsis});

  return sliced_output;
}

TcnnEncoding::TcnnEncoding(
    const torch::DeviceType device,
    int input_dims,
    const nlohmann::json& config,
    const char* params_name)
    : TcnnModule(
          device,
          tcnn::cpp::create_encoding(input_dims, config, tcnn::cpp::Precision::Fp32),
          params_name)
{}

#if !defined(TCNN_NO_NETWORKS)
TcnnNetwork::TcnnNetwork(
    const torch::DeviceType device,
    int input_dims,
    int output_dims,
    const nlohmann::json& config,
    const char* params_name)
    : TcnnModule(
          device,
          tcnn::cpp::create_network(input_dims, output_dims, config),
          params_name),
      output_dims(output_dims)
{}
#endif // !defined(TCNN_NO_NETWORKS)

}}  // end namespace p11::nerf
