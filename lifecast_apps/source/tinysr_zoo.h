// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <string>
#include "torch/torch.h"
#include "logger.h"

namespace p11 { namespace enhance {

struct Base_SuperResModel : torch::nn::Module {
    virtual torch::Tensor forward(torch::Tensor x) = 0;
    virtual ~Base_SuperResModel() = default;
};


struct Nano_SuperResModelImpl : Base_SuperResModel {
  static constexpr int k = 32;
  torch::nn::Conv2d conv_in{nullptr}, conv_mid{nullptr}, conv_out{nullptr};
  torch::nn::PixelShuffle pixel_shuffle{nullptr};

  Nano_SuperResModelImpl() {
    conv_in = register_module("conv_in", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, k, 5).padding(2)));
    conv_mid = register_module("conv_mid", torch::nn::Conv2d(torch::nn::Conv2dOptions(k, k * 4, 3).padding(1)));
    conv_out = register_module("conv_out", torch::nn::Conv2d(torch::nn::Conv2dOptions(k, 3, 5).padding(2)));
    pixel_shuffle = register_module("pixel_shuffle", torch::nn::PixelShuffle(2));
  }

  torch::Tensor forward(torch::Tensor x_in) override {
    auto x = torch::relu(conv_in->forward(x_in));
    x = torch::relu(conv_mid->forward(x));
    x = pixel_shuffle->forward(x);
    x = conv_out->forward(x);
    return x;
  }
};
TORCH_MODULE(Nano_SuperResModel);


struct ResidualBlockImpl : Base_SuperResModel {
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};

  ResidualBlockImpl(int channels) {
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
  }

  torch::Tensor forward(torch::Tensor x) {
    auto residual = x;
    x = torch::relu(conv1->forward(x));
    x = conv2->forward(x);
    //x = 0.1 * (x + residual); // Bad
    x = x * 0.1 + residual;     // Huge improvement
    return x;
  }
};
TORCH_MODULE(ResidualBlock);


struct EDSR_SuperResModelImpl : Base_SuperResModel {
  int k, num_res_blocks;
  torch::nn::Conv2d conv_in{nullptr}, conv_mid{nullptr}, conv_out{nullptr};
  //torch::nn::Upsample upsample{nullptr};
  torch::nn::PixelShuffle pixel_shuffle{nullptr};
  torch::nn::ModuleList res_blocks{nullptr};

  EDSR_SuperResModelImpl(int k, int num_res_blocks) : k(k), num_res_blocks(num_res_blocks) {
    conv_in = register_module("conv_in", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, k, 5).padding(2)));
    res_blocks = register_module("res_blocks", torch::nn::ModuleList());
    for (int i = 0; i < num_res_blocks; ++i) {
      res_blocks->push_back(ResidualBlock(k));
    }
    conv_mid = register_module("conv_mid", torch::nn::Conv2d(torch::nn::Conv2dOptions(k, k * 4, 3).padding(1)));
    pixel_shuffle = register_module("pixel_shuffle", torch::nn::PixelShuffle(2));
    conv_out = register_module("conv_out", torch::nn::Conv2d(torch::nn::Conv2dOptions(k, 3, 5).padding(2)));
    //upsample = register_module("upsample", torch::nn::Upsample(torch::nn::UpsampleOptions()
    //  .scale_factor(std::vector<double>{2.0, 2.0})
    //  .mode(torch::kBilinear)
    //  .align_corners(false)));
  }

  torch::Tensor forward(torch::Tensor x_in) override {
    //auto x0 = upsample->forward(x_in);
    auto x = torch::relu(conv_in->forward(x_in));
    //auto residual = x; // Long range residual
    for (const auto& block : *res_blocks) {
      x = block->as<ResidualBlock>()->forward(x);
    }
    //x = x + residual; // Long range residual
    x = torch::relu(conv_mid->forward(x));
    x = pixel_shuffle->forward(x);
    x = conv_out->forward(x);
    //return x0  + x;
    return x;
  }
};
TORCH_MODULE(EDSR_SuperResModel);


struct UNet_SuperResModelImpl : Base_SuperResModel {
  int k, num_res_blocks;
  torch::nn::Conv2d conv_in{nullptr}, conv_mid{nullptr};//, conv_out{nullptr};
  torch::nn::Conv2d conv_down1{nullptr}, conv_down2{nullptr}, conv_enc1{nullptr}, conv_enc2{nullptr}, conv_dec1{nullptr}, conv_dec2{nullptr};
  torch::nn::ConvTranspose2d conv_up1{nullptr}, conv_up2{nullptr};
  torch::nn::PixelShuffle pixel_shuffle{nullptr};
  torch::nn::ModuleList res_blocks{nullptr};
  //torch::nn::Upsample upsample{nullptr};

  UNet_SuperResModelImpl(int k, int num_res_blocks) : k(k), num_res_blocks(num_res_blocks) {
    conv_in = register_module("conv_in",        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, k, 3).padding(1)));
    conv_down1 = register_module("conv_down1",  torch::nn::Conv2d(torch::nn::Conv2dOptions(k, k, 3).stride(2).padding(1)));
    conv_down2 = register_module("conv_down2",  torch::nn::Conv2d(torch::nn::Conv2dOptions(k, k, 3).stride(2).padding(1)));
    conv_up1 = register_module("conv_up1",      torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(k, k, 3).stride(2).padding(1).output_padding(1)));
    conv_up2 = register_module("conv_up2",      torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(k, k, 3).stride(2).padding(1).output_padding(1)));
    conv_enc1 = register_module("conv_enc1",    torch::nn::Conv2d(torch::nn::Conv2dOptions(k, k, 3).padding(1)));
    conv_dec1 = register_module("conv_dec1",    torch::nn::Conv2d(torch::nn::Conv2dOptions(k, k, 3).padding(1)));
    conv_enc2 = register_module("conv_enc2",    torch::nn::Conv2d(torch::nn::Conv2dOptions(k, k, 3).padding(1)));
    conv_dec2 = register_module("conv_dec2",    torch::nn::Conv2d(torch::nn::Conv2dOptions(k, k, 5).padding(2)));
    conv_mid = register_module("conv_mid",      torch::nn::Conv2d(torch::nn::Conv2dOptions(k, 3 * 4, 3).padding(1)));
    //conv_out = register_module("conv_out",      torch::nn::Conv2d(torch::nn::Conv2dOptions(k, 3, 5).padding(2)));

    res_blocks = register_module("res_blocks", torch::nn::ModuleList());
    for (int i = 0; i < num_res_blocks; ++i) {
      res_blocks->push_back(ResidualBlock(k));
    }
    
    pixel_shuffle = register_module("pixel_shuffle", torch::nn::PixelShuffle(2));

    //upsample = register_module("upsample", torch::nn::Upsample(torch::nn::UpsampleOptions()
    //  .scale_factor(std::vector<double>{2.0, 2.0})
    //  .mode(torch::kBilinear)
    //  .align_corners(false)));
  }

  torch::Tensor forward(torch::Tensor x_in) {
    //auto x0 = upsample->forward(x_in);
    auto x = torch::relu(conv_in->forward(x_in)); // initial convolution to get from 3 -> K channels
    x = torch::relu(conv_enc1->forward(x));       // encoder conv 1
    auto residual1 = x;                           // long range residual (full size)
    x = torch::relu(conv_down1->forward(x));      // down to 1/2 size
    x = torch::relu(conv_enc2->forward(x));       // encoder conv 2
    auto residual2 = x;                           // long range residual (half size)
    x = torch::relu(conv_down2->forward(x));      // down to 1/4 size
    for (const auto& block : *res_blocks) {       
      x = block->as<ResidualBlock>()->forward(x); // resnet backbone
    }
    x = torch::relu(conv_up1->forward(x));        // up to 1/2 size
    x = torch::relu(conv_dec1->forward(x));       // decoder conv 1
    x = x + residual2;                            // long range residual (half size)
    x = torch::relu(conv_up2->forward(x));        // up to 1x (input) size
    x = torch::relu(conv_dec2->forward(x));       // decoder conv 2
    x = x + residual1;                            // long range residual (full size)
    x = torch::relu(conv_mid->forward(x));        // convolve up to the number of channels need for pixel_shuffle
    x = pixel_shuffle->forward(x);                // up to 2x (super res)
    //x = conv_out->forward(x);                     // do one more convolution to clean up pixel shuffle artifacts
    return x;
    //return x0 + x;                              // predict residual from upscaling
  }
};
TORCH_MODULE(UNet_SuperResModel);


inline std::shared_ptr<Base_SuperResModel> makeSuperResModelByName(const std::string& name) {
  if (name == "nano") {   return std::make_shared<Nano_SuperResModel>()->ptr(); }
  if (name == "edsr") {   return std::make_shared<EDSR_SuperResModel>(32, 4)->ptr(); } // channels, num_res_blocks
  if (name == "unet") {   return std::make_shared<UNet_SuperResModel>(64, 1)->ptr(); } // channels, num_res_blocks
  if (name == "edsr2") {  return std::make_shared<EDSR_SuperResModel>(64, 8)->ptr(); } // channels, num_res_blocks
  if (name == "unet2") {  return std::make_shared<UNet_SuperResModel>(64, 8)->ptr(); } // channels, num_res_blocks
  if (name == "edsr3") {  return std::make_shared<EDSR_SuperResModel>(64, 16)->ptr(); } // channels, num_res_blocks
  if (name == "unet3") {  return std::make_shared<UNet_SuperResModel>(64, 16)->ptr(); } // channels, num_res_blocks
  XCHECK(false) << "Unknown model: " << name;
  return nullptr;
}

}}  // end namespace p11::enhance
