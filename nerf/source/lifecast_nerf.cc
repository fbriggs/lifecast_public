// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "gflags/gflags.h"
#include "logger.h"
#include "check.h"
#include "util_file.h"
#include "lifecast_nerf_lib.h"

#ifdef _WIN32
#include <Windows.h>
#endif

DEFINE_string(train_dir,        "", "auto-fills train_images_dir and train_json based on a particular directory structure");
DEFINE_string(train_images_dir, "", "path to training image directory");
DEFINE_string(train_json,       "", "path to training dataset json");
DEFINE_string(output_dir,       "", "path to write outputs");
DEFINE_string(load_model_dir,   "", "if non-empty, load models from this folder at the start (of training, or to just immediately render novel views)");
DEFINE_string(vid_dir,          "", "if this is not empty, we will run in video mode instead using this as the source directory");

DEFINE_int32(num_training_itrs, 5000, "number of iterations for training");
DEFINE_int32(rays_per_batch,    4096, "number of rays per batch in training");
DEFINE_int32(num_basic_samples, 128, "number of samples per ray using uniform and/or inverse distance sampling");
DEFINE_int32(num_importance_samples, 64, "number of points per ray sampled using importance sampling");
DEFINE_int32(warmup_itrs,       100, "number of iterations to go before doing any importance sampling");
DEFINE_int32(num_novel_views,   0, "number of novel view images to render");
DEFINE_bool(compute_train_psnr, true, "whether or not to compute PSNR on the training images (which is slow)");
// Learning rates and optimizer parameters
DEFINE_double(radiance_lr,      1e-2, "learning rate for radiance model");
DEFINE_double(radiance_decay,   1e-8, "weight decay for radiance model");
DEFINE_double(image_code_lr,    1e-4, "learning rate for per-image embeddings");
DEFINE_double(image_code_decay, 1e-4, "weight decay for per-image embeddings");
DEFINE_double(prop_lr,          1e-2, "learning rate for proposal model");
DEFINE_double(prop_decay,       1e-6, "weight decay for proposal model");
DEFINE_double(adam_eps,         1e-17, "eps parameter for Adam optimizer");
// Regularization parameters
DEFINE_double(floater_min_dist, 1.0, "regularizer penalizes density closer than this from camera. note this is unitless");
DEFINE_double(floater_weight,   1e-3, "weight for floater regularization");
DEFINE_double(gini_weight,      1e-5, "weight for info (gini) regularization");
DEFINE_double(distortion_weight,3e-2, "weight for distortion loss");
DEFINE_double(density_weight,   1e-6, "weight of penalty on any density in the radiance field");
DEFINE_double(visibility_weight, 1e-4, "weight for visibility regularization");
DEFINE_int32(num_visibility_points, 1024, "number of points to sample per batch for visibility regularization");
// Different mode; distill (not train)
DEFINE_bool(distill_ldi3,        false, "If true, the regular training pipeline is not used. Instead, load an existing model and distill it into an LDI3.");
DEFINE_string(distill_model_dir, "", "Path to folder containing model files to distill into LDI3.");
DEFINE_bool(transparent_bg,      false, "Whether to generate a transparent background layer when baking an LDI.");
// Video pipeline options
DEFINE_int32(ldi_resolution,       1920, "Resolution of one cell of LDI output");
DEFINE_double(prev_density_weight, 1e-4, "weight of regularization toward previous radiance field for temporal stability");
DEFINE_int32(prev_reg_num_samples, 32, "number of samples per ray regularization toward previous radiance field for temporal stablity");

int main(int argc, char** argv)
{
// Workaround a bug in libtorch for windows where it links against the wrong dll and doesn't support
// CUDA. See https://github.com/pytorch/pytorch/issues/72396
#ifdef _WIN32
  LoadLibraryA("torch_cuda.dll");
#endif

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_distill_ldi3) {
    p11::nerf::testNerfToLdi3Distillation(FLAGS_distill_model_dir, FLAGS_output_dir, FLAGS_ldi_resolution, FLAGS_transparent_bg);
    return EXIT_SUCCESS;
  }

  XCHECK(!FLAGS_output_dir.empty());
  p11::file::createDirectoryIfNotExists(FLAGS_output_dir);
  XCHECK(p11::file::directoryExists(FLAGS_output_dir));

  //p11::nerf::testImportanceSampling(); return EXIT_SUCCESS;

  // This is just some syntactic sugar on how we can specify the inputs
  std::string train_images_dir = FLAGS_train_images_dir;
  std::string train_json = FLAGS_train_json;
  if (!FLAGS_train_dir.empty()) {
    train_images_dir = FLAGS_train_dir;
    train_json = FLAGS_train_dir + "/dataset_train.json";
  }

  p11::nerf::NeoNerfConfig cfg;
  cfg.cancel_requested = nullptr;
  cfg.train_images_dir = train_images_dir;
  cfg.train_json = train_json;
  cfg.output_dir = FLAGS_output_dir;
  cfg.load_model_dir = FLAGS_load_model_dir;
  cfg.num_training_itrs = FLAGS_num_training_itrs;
  cfg.rays_per_batch = FLAGS_rays_per_batch;
  cfg.num_basic_samples = FLAGS_num_basic_samples;
  cfg.num_importance_samples = FLAGS_num_importance_samples;
  cfg.warmup_itrs = FLAGS_warmup_itrs;
  cfg.num_novel_views = FLAGS_num_novel_views;
  cfg.compute_train_psnr = FLAGS_compute_train_psnr;
  cfg.radiance_lr = FLAGS_radiance_lr;
  cfg.radiance_decay = FLAGS_radiance_decay;
  cfg.image_code_lr = FLAGS_image_code_lr;
  cfg.image_code_decay = FLAGS_image_code_decay;
  cfg.prop_lr = FLAGS_prop_lr;
  cfg.prop_decay = FLAGS_prop_decay;
  cfg.adam_eps = FLAGS_adam_eps;
  cfg.floater_min_dist = FLAGS_floater_min_dist;
  cfg.floater_weight = FLAGS_floater_weight;
  cfg.gini_weight = FLAGS_gini_weight;
  cfg.distortion_weight = FLAGS_distortion_weight;
  cfg.density_weight = FLAGS_density_weight;
  cfg.visibility_weight = FLAGS_visibility_weight;
  cfg.num_visibility_points = FLAGS_num_visibility_points;  
  cfg.prev_density_weight = FLAGS_prev_density_weight;
  cfg.prev_reg_num_samples = FLAGS_prev_reg_num_samples;

  if (FLAGS_vid_dir.empty()) {
    XCHECK(!FLAGS_train_dir.empty() || (!FLAGS_train_images_dir.empty() && !FLAGS_train_json.empty()));
    p11::nerf::runNerfPipeline(cfg);
  } else {
    p11::nerf::NerfVideoConfig vid_cfg;
    vid_cfg.vid_dir = FLAGS_vid_dir;
    vid_cfg.ldi_resolution = FLAGS_ldi_resolution;
    vid_cfg.transparent_bg = FLAGS_transparent_bg;
    p11::nerf::runVideoNerfPipeline(vid_cfg, cfg);
  }
  return EXIT_SUCCESS;
}
