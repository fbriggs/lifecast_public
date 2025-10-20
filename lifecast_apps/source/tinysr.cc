// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
== Training ===

1. Train a model with no augmentation

bazel run -c opt -- //source:tinysr \
--mode="train" \
--model_name="edsr2" \
--augment="none" \
--train_images_dir ~/Downloads/df2k_png_curated \
--test_images_dir  ~/Downloads/df2k_hr_jpg_test \
--dest_dir ~/Desktop/super_res \
--num_itrs 250000

OR (for unet model):

bazel run -c opt -- //source:tinysr \
--mode="train" \
--model_name="unet" \
--augment="none" \
--train_images_dir ~/Downloads/df2k_png_curated \
--test_images_dir  ~/Downloads/df2k_hr_jpg_test \
--dest_dir ~/Desktop/super_res \
--num_itrs 250000

2. Refine the model with some augmentation (rename edsr2_noaug.pt as needed)

bazel run -c opt -- //source:tinysr \
--mode="train" \
--model_name="edsr2" \
--augment="mystery" \
--model_file ~/Desktop/super_res/edsr2_noaug.pt \
--train_images_dir ~/Downloads/df2k_png_curated \
--test_images_dir  ~/Downloads/df2k_hr_jpg_test \
--dest_dir ~/Desktop/super_res \
--num_itrs 250000

OR

bazel run -c opt -- //source:tinysr \
--mode="train" \
--model_name="unet" \
--augment="mystery" \
--model_file ~/Desktop/super_res/unet_noaug.pt \
--train_images_dir ~/Downloads/df2k_png_curated \
--test_images_dir  ~/Downloads/df2k_hr_jpg_test \
--dest_dir ~/Desktop/super_res \
--num_itrs 250000

NOTE: You can also get the pre-trained model from the repo, something like
--model_file ~/dev/p11/ml_models/edsr2_noaug_cpu.pt \

== Run the model ===

There should generally be a model file named super_resolution_model.pt (regardless of which model_name we train)

bazel run -c opt -- //source:tinysr \
--mode="upscale" \
--model_name="edsr2" \
--model_file ~/Desktop/super_res/super_resolution_model.pt \
--dest_image ~/Desktop/super_res/test.png \
--src_image ~/Downloads/df2k_hr_jpg_test/0004.jpg

OR refer to the model file by name:

bazel run -c opt -- //source:tinysr \
--mode="upscale" \
--model_name="edsr2" \
--model_file ~/Desktop/super_res/edsr2_cpu.pt \
--dest_image ~/Desktop/super_res/test.png \
--src_image ~/Downloads/df2k_hr_jpg_test/0004.jpg

4k x 4k input test file:
--src_image ~/Desktop/super_res_training_images/waves.png \

== Run the full PSNR eval for a pre-trained model ==

bazel run -c opt -- //source:tinysr \
--mode="eval" \
--model_name="edsr2" \
--model_file ~/dev/p11/ml_models/edsr2_noaug.pt \
--augment="none" \
--test_images_dir  ~/Downloads/df2k_hr_jpg_test
*/

#include "gflags/gflags.h"
#include "logger.h"
#include "tinysr_lib.h"

#ifdef _WIN32
#include <Windows.h>
#endif

DEFINE_int32(rng_seed, 123, "seed for rng");
DEFINE_string(mode, "train", "train,upscale,eval");
DEFINE_string(model_name, "edsr", "nano,edsr,unet");
DEFINE_double(scale, 2, "amount of scaling");
DEFINE_int32(batch_size, 32, "batch size during training");
DEFINE_double(lr, 0.001, "learning rate");
DEFINE_double(lr_decay, 0.333333, "learning rate decay");
// Training parameters
DEFINE_string(train_images_dir, "", "path to folder containing images for training");
DEFINE_string(test_images_dir, "", "path to folder containing images for periodic testing during training");
DEFINE_string(dest_dir, "", "path to write outputs");
DEFINE_string(augment, "none", "none,gblur,mystery");
DEFINE_int32(num_itrs, 100000, "number of training iterations");
// Inference (mode="upscale")
DEFINE_string(model_file, "", "path to trained model .pt file");
DEFINE_string(src_image, "", "path to input image");
DEFINE_string(dest_image, "", "path to write output");

int main(int argc, char** argv)
{
// Workaround a bug in libtorch for windows where it links against the wrong dll and doesn't support
// CUDA. See https://github.com/pytorch/pytorch/issues/72396
#ifdef _WIN32
  LoadLibraryA("torch_cuda.dll");
#endif
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  p11::enhance::TinySuperResConfig cfg;
  cfg.rng_seed = FLAGS_rng_seed;
  cfg.mode = FLAGS_mode;
  cfg.model_name = FLAGS_model_name;
  cfg.scale = FLAGS_scale;
  cfg.batch_size = FLAGS_batch_size;
  cfg.lr = FLAGS_lr;
  cfg.lr_decay = FLAGS_lr_decay;
  cfg.train_images_dir = FLAGS_train_images_dir;
  cfg.test_images_dir = FLAGS_test_images_dir;
  cfg.dest_dir = FLAGS_dest_dir;
  cfg.augment = FLAGS_augment;
  cfg.num_itrs = FLAGS_num_itrs;
  cfg.model_file = FLAGS_model_file;
  cfg.src_image = FLAGS_src_image;
  cfg.dest_image = FLAGS_dest_image;

  if (FLAGS_mode == "train") {
    XCHECK(!FLAGS_train_images_dir.empty());
    XCHECK(!FLAGS_dest_dir.empty());
    p11::enhance::trainSuperResModel(cfg);
  }

  if (FLAGS_mode == "upscale") {
    XCHECK(!FLAGS_model_file.empty());
    XCHECK(!FLAGS_src_image.empty());
    XCHECK(!FLAGS_dest_image.empty());
    p11::enhance::testSuperResModel(cfg);
  }

  if (FLAGS_mode == "eval") {
    XCHECK(!FLAGS_model_file.empty());
    XCHECK(!FLAGS_test_images_dir.empty());
    XCHECK(!FLAGS_model_file.empty());
    p11::enhance::loadAndEvalSuperResModel(cfg);
  }

  return EXIT_SUCCESS;
}
