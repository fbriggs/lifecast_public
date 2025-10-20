// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "tinysr_lib.h"

#include "util_string.h"
#include "util_file.h"
#include "util_torch.h"
#include "util_time.h"
#include "util_math.h"
#include "util_opencv.h"
#include "torch_opencv.h"

namespace p11 { namespace enhance {

std::vector<std::string> getImageFilenames(const std::string& image_dir) {
  std::vector<std::string> image_paths_raw = file::getFilesInDir(image_dir);
  std::vector<std::string> image_paths;
  for (std::string& filename : image_paths_raw) {
    const std::string ext = file::filenameExtension(filename);
    if (ext == "png" || ext == "jpg" || ext == "PNG" || ext == "JPG") {
      image_paths.push_back(filename);
    }
  }
  return image_paths;
}

void augmentImage(const std::string mode, cv::Mat& image) {
  if (mode == "none") return;

  if (mode == "gblur") {
    cv::GaussianBlur(image, image, cv::Size(5, 5), 0.5, 0.5);
  }

  if (mode == "mystery") {
    int original_type = image.type();
    opencv::convertToWithAutoScale(image, image, CV_32FC3);

    cv::Mat noise_image(image.size(), CV_32FC3);
    float sigma_noise = math::randUnif() * 0.03;
    cv::randn(noise_image, 0, sigma_noise);
    float noise_blur_radius = math::randUnif() * 0.3;
    cv::GaussianBlur(noise_image, noise_image, cv::Size(7, 7), noise_blur_radius, noise_blur_radius);
    image += noise_image;
    cv::max(image, 0, image);
    cv::min(image, 1, image);
    float blur_radius = math::randUnif() * 0.6;
    cv::GaussianBlur(image, image, cv::Size(7, 7), blur_radius, blur_radius);

    opencv::convertToWithAutoScale(image, image, original_type);
  }

  //cv::imshow("augmented", image); cv::waitKey(0);
}

void saveCpuModelCopy(std::shared_ptr<Base_SuperResModel> model, const TinySuperResConfig& cfg) {
  auto model_copy_cpu = makeSuperResModelByName(cfg.model_name);
  util_torch::deepCopyModel(model, model_copy_cpu);
  model_copy_cpu->to(torch::kCPU);
  torch::save(model_copy_cpu, cfg.dest_dir + "/" + cfg.model_name + "_cpu.pt");
}

void trainSuperResModel(const TinySuperResConfig& cfg) {
  torch::manual_seed(cfg.rng_seed);  // For reproducible initialization of weights
  srand(cfg.rng_seed);               // For calls to rand()
  const torch::DeviceType device = util_torch::findBestTorchDevice();
  // Encourage deterministic behavior with CUDA
#if defined(__linux__) || defined(_WIN32)  
  if (device == torch::kCUDA) {
    at::globalContext().setDeterministicCuDNN(true);
    at::globalContext().setDeterministicAlgorithms(true, false);
  }
#endif

  const std::vector<std::string> train_image_filenames = getImageFilenames(cfg.train_images_dir);
  const int num_train_images = train_image_filenames.size();

  auto model = makeSuperResModelByName(cfg.model_name);
  if (!cfg.model_file.empty()) {
    XPLINFO << "Loading pretrained model: " << cfg.model_file;
    torch::load(model, cfg.model_file);
  }
  model->to(device);
  model->eval();


  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(cfg.lr));

  double running_psnr = 0;

  float loss_moving_avg = 0;
  torch::Tensor loss = torch::zeros({1}, torch::kFloat32).to(device);

  // TODO: adjust these to be better for SR. maybe once per 100k itrs?
  std::vector<int> milestones = {
    75000,
    150000,
    300000,
    400000,
    500000
  };

  std::ofstream f_log(cfg.dest_dir + "/train_" + cfg.model_name + ".txt", std::ios::out);
  for (int itr = 0; itr < cfg.num_itrs; ++itr) {
    applyLearningRateSchedule(optimizer, cfg.lr, itr, milestones, cfg.lr_decay);
    //auto timer = time::now();
    const int rand_img = rand() % num_train_images;
    std::string image_filename = train_image_filenames[rand_img];
    cv::Mat image_hr = cv::imread(cfg.train_images_dir + "/" + image_filename);

    // Crop a random patch from the image
    constexpr int kCropSize = 512;
    int max_x = std::max(0, image_hr.cols - kCropSize);
    int max_y = std::max(0, image_hr.rows - kCropSize);
    int x = rand() % (max_x + 1);
    int y = rand() % (max_y + 1);
    cv::Rect roi(x, y, std::min(512, image_hr.cols - x), std::min(kCropSize, image_hr.rows - y));
    image_hr = image_hr(roi);
    //cv::imshow("image_hr", image_hr); cv::waitKey(1);

    // Make sure the image size is divisible by 2
    //cv::Rect roi(0, 0, image_hr.cols & ~1, image_hr.rows & ~1);
    //image_hr = image_hr(roi);

    // Make the low-res image
    cv::Mat image_lr;
    cv::resize(image_hr, image_lr, cv::Size(image_hr.cols / cfg.scale, image_hr.rows / cfg.scale), 0, 0, cv::INTER_AREA);
    
    // Corrupt the low resolution image
    augmentImage(cfg.augment, image_lr);

    torch::Tensor tensor_hr = torch_opencv::cvMat_to_Tensor(device, image_hr);
    torch::Tensor tensor_lr = torch_opencv::cvMat_to_Tensor(device, image_lr);
    torch::Tensor predicted_hr = model->forward(tensor_lr.unsqueeze(0)).squeeze(0);

    //loss += torch::mse_loss(predicted_hr, tensor_hr);
    loss += torch::l1_loss(predicted_hr, tensor_hr); // TODO: does this help?

    if (itr % cfg.batch_size == (cfg.batch_size - 1)) {
      loss.backward();
      optimizer.step();
      optimizer.zero_grad();
      loss /= cfg.batch_size;
      XPLINFO << itr << "\t" << loss.item<float>() << "\tpsnr: " << running_psnr;

      loss_moving_avg = (itr == 0)
        ? loss.item<float>()
        : loss_moving_avg * 0.9 + loss.item<float>() * 0.1;
      f_log << itr << "\t" << std::log10(loss_moving_avg) << std::endl;
      loss = torch::zeros({1}, torch::kFloat32).to(device);
    }

    if (itr % 1000 == 0 || itr == cfg.num_itrs - 1) {
      running_psnr = evalSuperResModel(cfg, model);
      torch::save(model, cfg.dest_dir + "/super_resolution_model.pt");
      saveCpuModelCopy(model, cfg);
    }
    //XPLINFO << "itr time: " << time::timeSinceSec(timer);
  }
  f_log.close();
}

double applyLearningRateSchedule(
  torch::optim::Optimizer& optimizer,
  double initial_lr,
  int current_iter,
  const std::vector<int>& milestones,
  double gamma // Learning rate decays by this factor each time we reach a milestone
){
  // Apply LR decay at each milestone
  double curr_lr = initial_lr;
  for (int milestone : milestones) {
    if (current_iter >= milestone) {
      curr_lr *= gamma;
    }
  }

  // Update the learning rate for each parameter group
  for (auto& group : optimizer.param_groups()) {
    auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
    options.lr(curr_lr);
  }

  return curr_lr;
}

// We run the SR model on very high res inputs, which can use prohibitive amounts of memory..
// NOTE: Hard-coded to a 2x scale factor
torch::Tensor tiledInference(
  const torch::Tensor& input,
  std::shared_ptr<Base_SuperResModel> model,
  int tile_size,
  int overlap
) {
  int scale_factor = 2;
  int channels = input.size(0);
  int input_height = input.size(1);
  int input_width = input.size(2);
  int output_height = input_height * scale_factor;
  int output_width = input_width * scale_factor;
  torch::Tensor output = torch::zeros({channels, output_height, output_width}, input.options());
  torch::Tensor weight = torch::zeros({1, output_height, output_width}, input.options());

  for (int y = 0; y < output_height; y += tile_size - overlap) {
    for (int x = 0; x < output_width; x += tile_size - overlap) {
      int end_y = std::min(y + tile_size, output_height);
      int end_x = std::min(x + tile_size, output_width);
      int start_y = std::max(0, end_y - tile_size);
      int start_x = std::max(0, end_x - tile_size);

      torch::Tensor input_tile = input.slice(1, start_y / scale_factor, end_y / scale_factor).slice(2, start_x / scale_factor, end_x / scale_factor);
      torch::Tensor output_tile = model->forward(input_tile.unsqueeze(0)).squeeze(0);

      // Create a weight mask for smooth blending
      torch::Tensor mask = torch::ones({1, end_y - start_y, end_x - start_x}, input.options());
      for (int i = 0; i < overlap; ++i) {
        float w = 0.5f * (1 - std::cos(M_PI * i / overlap));
        if (start_y > 0) mask.slice(1, i, i+1).fill_(w);
        if (end_y < output_height) mask.slice(1, -i-1, -i).fill_(w);
        if (start_x > 0) mask.slice(2, i, i+1).fill_(w);
        if (end_x < output_width) mask.slice(2, -i-1, -i).fill_(w);
      }

      output.slice(1, start_y, end_y).slice(2, start_x, end_x).add_(output_tile * mask);
      weight.slice(1, start_y, end_y).slice(2, start_x, end_x).add_(mask);
    }
  }

  // Normalize the output by the accumulated weights
  output.div_(weight.add(1e-8));
  return output;
}

torch::Tensor superResolutionEnhance(
  const torch::DeviceType device,
  float scale,
  std::shared_ptr<Base_SuperResModel> model,
  cv::Mat input_image
) {
  XCHECK(input_image.type() == CV_8UC3 || input_image.type() == CV_16UC3 || input_image.type() == CV_32FC3);

  torch::NoGradGuard no_grad;

  // Make sure the image size is divisible by 2
  cv::Rect roi(0, 0, input_image.cols & ~1, input_image.rows & ~1);
  input_image = input_image(roi);

  torch::Tensor input_tensor = torch_opencv::cvMat_to_Tensor(device, input_image);

  int tile_size = 512; 
  int overlap = 32;
  //auto forward_timer = time::now();
  torch::Tensor output_tensor = tiledInference(input_tensor, model, tile_size, overlap);
  //torch::Tensor output_tensor = model->forward(input_tensor.unsqueeze(0)).squeeze(0);
  //XPLINFO << "inference time: " << time::timeSinceSec(forward_timer);

  output_tensor = output_tensor.permute({1, 2, 0});  
  output_tensor = output_tensor.clamp(0, 1);
  return output_tensor;
}

void superResolutionEnhance(
  const torch::DeviceType device,
  float scale,
  std::shared_ptr<Base_SuperResModel> model,
  cv::Mat input_image,
  cv::Mat& output_image,
  cv::Mat& image_bicubic_upscaled,
  int cv_output_type
) {
  XCHECK(cv_output_type == CV_8UC3 || cv_output_type == CV_32FC3);
  
  cv::resize(input_image, image_bicubic_upscaled, cv::Size(input_image.cols * scale, input_image.rows * scale), 0, 0, cv::INTER_CUBIC);

  torch::Tensor output_tensor = superResolutionEnhance(device, scale, model, input_image);
  
  //auto unpack_time = time::now();
  torch_opencv::fastTensor_To_CvMat(output_tensor, output_image);

  //torch::cuda::synchronize();
  //XPLINFO << "unpack time: " << time::timeSinceSec(unpack_time);

  if (cv_output_type == CV_8UC3) {
    output_image.convertTo(output_image, CV_8UC3, 255.0f);
    image_bicubic_upscaled.convertTo(image_bicubic_upscaled, CV_8UC3);
  }
}

double evalSuperResModel(
  const TinySuperResConfig& cfg,
  std::shared_ptr<Base_SuperResModel> model
) {
  torch::NoGradGuard no_grad;

  const std::vector<std::string> test_image_filenames = getImageFilenames(cfg.test_images_dir);
  const int num_test_images = test_image_filenames.size();
  if (num_test_images == 0) {
    //XPLINFO << "Skipping test, No test images found in " << cfg.test_images_dir;
    return 0;
  }

  const torch::DeviceType device = util_torch::findBestTorchDevice();

  double ours_psnr_sum = 0;
  double bicubic_psnr_sum = 0;

  for (int i = 0; i < num_test_images; i++) {
    std::string image_filename = test_image_filenames[i];
    XPLINFO << "Opening image: " << cfg.test_images_dir + "/" + image_filename;
    cv::Mat image_hr = cv::imread(cfg.test_images_dir + "/" + image_filename);

    cv::Mat image_lr;
    cv::resize(image_hr, image_lr, cv::Size(image_hr.cols / cfg.scale, image_hr.rows / cfg.scale), 0, 0, cv::INTER_AREA);
    augmentImage(cfg.augment, image_lr); // For consistenecy with training

    int tile_size = 512; 
    int overlap = 32;
    torch::Tensor tensor_lr = torch_opencv::cvMat_to_Tensor(device, image_lr);
    torch::Tensor tensor_hr = torch_opencv::cvMat_to_Tensor(device, image_hr);
    torch::Tensor predicted_hr = tiledInference(tensor_lr, model, tile_size, overlap);
    cv::Mat bicubic_hr = cv::Mat(image_hr.rows, image_hr.cols, CV_8UC3);
    cv::resize(image_lr, bicubic_hr, bicubic_hr.size(), 0, 0, cv::INTER_CUBIC);
    torch::Tensor tensor_bicubic_hr = torch_opencv::cvMat_to_Tensor(device, bicubic_hr);

    torch::Tensor loss = torch::mse_loss(predicted_hr, tensor_hr);
    const double psnr = 20.0 * std::log10(1.0 / std::sqrt(loss.item<float>()));
    ours_psnr_sum += psnr;

    torch::Tensor bicubic_loss = torch::mse_loss(tensor_bicubic_hr, tensor_hr);
    const double bicubic_psnr = 20.0 * std::log10(1.0 / std::sqrt(bicubic_loss.item<float>()));
    bicubic_psnr_sum += bicubic_psnr;

    XPLINFO << "Image: " << image_filename << " PSNR: " << psnr << " baseline bicubic PSNR: " << bicubic_psnr;
  }

  const double avg_psnr = ours_psnr_sum / num_test_images;
  XPLINFO << "Average PSNR over " << num_test_images << " images: " << avg_psnr;
  const double avg_bicubic_psnr = bicubic_psnr_sum / num_test_images;
  XPLINFO << "Average baseline bicubic PSNR over " << num_test_images << " images: " << avg_bicubic_psnr;
  return avg_psnr;
}

double loadAndEvalSuperResModel(const TinySuperResConfig& cfg) {
  torch::NoGradGuard no_grad;
  const torch::DeviceType device = util_torch::findBestTorchDevice();
  std::shared_ptr<Base_SuperResModel> model = makeSuperResModelByName(cfg.model_name);
  torch::load(model, cfg.model_file);
  model->to(device);
  model->eval();

  auto timer = time::now();
  double avg_psnr = evalSuperResModel(cfg, model);
  XPLINFO << "evalSuperResModel time: " << time::timeSinceSec(timer);
  return avg_psnr;
}

void testSuperResModel(const TinySuperResConfig& cfg) {
  torch::NoGradGuard no_grad;
  const torch::DeviceType device = util_torch::findBestTorchDevice();
  std::shared_ptr<Base_SuperResModel> model = makeSuperResModelByName(cfg.model_name);
  torch::load(model, cfg.model_file);
  model->to(device);
  model->eval();

  cv::Mat input_image = cv::imread(cfg.src_image);
  XCHECK(!input_image.empty()) << cfg.src_image;

  auto timer = time::now();
  cv::Mat output_image, image_bicubic_upscaled;
  superResolutionEnhance(device, cfg.scale, model, input_image, output_image, image_bicubic_upscaled);
  XPLINFO << "superResolutionEnhance time: " << time::timeSinceSec(timer);

  cv::imwrite(cfg.dest_image, output_image);
  cv::imwrite(cfg.dest_image + ".b.png", image_bicubic_upscaled);
}

}}  // end namespace p11::enahnce
