// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
NOTE: for Nvidia 3090, use compute_86, for 4090, use compute_89. See https://developer.nvidia.com/cuda-gpus

== First time building or if you see third_party/gsplat stuff being built ==

bazel build --local_cpu_resources=10 --jobs=1 --cuda_archs=compute_86 -- //third_party:gsplat_cuda

== static scenes ==

bazel run --local_cpu_resources=10 --jobs=1 --cuda_archs=compute_86 -c opt -- //source:lifecast_splat \
--train_dir ~/dev/p11_nerf_benchmark_v2/rope_swing \
--output_dir ~/Desktop/splatout

ffmpeg -y -framerate 59.94 \
-i ~/Desktop/splatout/trainsplat/%06d.png \
-c:v libx264 -preset slow -crf 0 -pix_fmt yuv420p -movflags faststart \
~/Desktop/splatout/splatvid.mp4

== GoPro video scenes ==

bazel run --local_cpu_resources=10 --jobs=1 --cuda_archs=compute_86 -c opt -- //source:lifecast_splat \
--vid_dir ~/Downloads/redbarn2_denoised \
--output_dir ~/Desktop/splatvid \
--resize_max_dim 768

bazel run --local_cpu_resources=10 --jobs=1 --cuda_archs=compute_86 -c opt -- //source:lifecast_splat \
--vid_dir ~/Downloads/radical \
--output_dir ~/Desktop/splatvid \
--resize_max_dim 768

bazel run --cuda_archs=compute_86 -c opt -- //source:lifecast_splat \
--vid_dir ~/Downloads/lets_dance_short \
--output_dir ~/Desktop/splatvid \
--init_with_monodepth=false \
--resize_max_dim 1280 \
--n 100000 --num_itrs 1000

ffmpeg -y -framerate 59.94 \
-i ~/Desktop/splatvid/splat_frames/%06d.png \
-c:v libx264 -preset slow -crf 0 -pix_fmt yuv420p -movflags faststart \
~/Desktop/splatvid/radical_splat_h264.mp4

ffmpeg -y -framerate 59.94 \
-i ~/Desktop/splatvid/splat_frames/%06d.png \
-c:v libx265 -preset slow -x265-params lossless=1 -pix_fmt yuv420p -movflags faststart \
-profile:v main -tag:v hvc1 \
~/Desktop/splatvid/radical_splat_h265.mp4

ffmpeg -y -framerate 59.94 \
-i ~/Desktop/splatvid/splat_frames/%06d.png \
-c:v libx265 -preset slow -crf 3 -pix_fmt yuv420p -movflags faststart \
-profile:v main -tag:v hvc1 \
~/Desktop/splatvid/radical_splat_h265.mp4

== audio mix ==

ffmpeg -y \
-i ~/Downloads/radical/camera_p.mp4 \
-vn -c:a aac -b:a 192k \
-af "volume=2" \
~/Desktop/splatvid/audio.aac

ffmpeg -y \
-i ~/Desktop/splatvid/radical_splat_h264.mp4 \
-i ~/Desktop/splatvid/audio.aac \
-c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest \
~/Desktop/splatvid/radical_splat_h264a.mp4

== tests ==

bazel run --cuda_archs=compute_86 -c opt -- //source:lifecast_splat \
--mode ply2png \
--src_ply ~/Downloads/splat_000000.ply \
--dest_png ~/Desktop/splatout/splats.png

bazel run --cuda_archs=compute_86 -c opt -- //source:lifecast_splat \
--mode render \
--src_ply ~/Downloads/splat_000000.ply

bazel run --cuda_archs=compute_86 -c opt -- //source:lifecast_splat \
--mode render \
--src_ply ~/Desktop/splatout/splats.png
*/
#include "gflags/gflags.h"
#include "logger.h"
#include "check.h"
#include "util_file.h"
#include "lifecast_splat_lib.h"
#include "lifecast_splat_io.h"
#include "lifecast_splat_math.h"
#include "lifecast_splat_population.h"
#include "util_torch.h"

#ifdef _WIN32
#include <Windows.h>
#endif

DEFINE_string(vid_dir,          "", "if this is not empty, we will run in video mode instead using this as the source directory");
DEFINE_string(train_dir,        "",     "auto-fills train_images_dir and train_json based on a particular directory structure");
DEFINE_string(sfm_pointcloud_path, "", "override default path if not empty to sfm point cloud bin file");
DEFINE_string(output_dir,       "",     "path to write outputs");
DEFINE_bool(save_steps,         false,  "if true, write the splats during every iteration of training to /trainsplat");
DEFINE_bool(calc_psnr,          false,  "if true, calculate PSNR on the whole training set");
DEFINE_int32(resize_max_dim,    0,      "If not 0, scale images so this is the larger of width or height (and scale intrinsics)");
DEFINE_int32(n,                 262144, "max_num_splats");
DEFINE_int32(num_itrs,          2000,   "number of training iterations");
DEFINE_int32(first_frame_warmup_itrs, 2000,   "number of training iterations on first frame of video");
DEFINE_int32(images_per_batch,  4,      "number of images batch"); // TODO: what happens if we select the same image multiple times?
DEFINE_int32(train_vis_interval, 0,     "iters between training visualization (0 to disable)");
DEFINE_int32(popi,              200,    "population dynamics update interval");
DEFINE_double(learning_rate,    3e-3,   "initial learning rate for adam optimizer before decay schedule");
DEFINE_bool(init_with_monodepth, false,  "if true, some of the initial splats are placed at skysphere distance");
DEFINE_bool(use_depth_loss,      false,  "include a depth loss function (e.g. based on mono depthmap)");

DEFINE_string(mode, "",  "which test to run, if not empty");
DEFINE_string(src_ply, "",  "the path to an input ply file");
DEFINE_string(src_png, "",  "the path to an input png file");
DEFINE_string(dest_png, "",  "path to write output ply when in convert mode");

void convertPlyToPng() {
  using namespace p11;
  using namespace p11::splat;
  const int kSplatImageWidth = 2048;
  const int kSplatImageHeight = 2048;
  std::vector<SerializableSplat> splats = loadSplatPLYFile(FLAGS_src_ply);
  cv::Mat splat_image = encodeSplatsInImage(splats, kSplatImageWidth, kSplatImageHeight);
  cv::imwrite(FLAGS_dest_png, splat_image);
}

void testUnboundContractExpand() {
  std::vector<Eigen::Vector3f> test_vecs = {
    Eigen::Vector3f(1, 2, 3),
    Eigen::Vector3f(10, 20, 30),
    Eigen::Vector3f(100, 2000, 30000) };
  for (Eigen::Vector3f& x : test_vecs) {
    Eigen::Vector3f c = p11::splat::contractUnbounded(x);
    Eigen::Vector3f x2 = p11::splat::expandUnbounded(c);
    XPLINFO << "x=" << x.x() << " " << x.y() << " " << x.z();
    XPLINFO << "c=" << c.x() << " " << c.y() << " " << c.z();
    XPLINFO << "x2=" << x2.x() << " " << x2.y() << " " << x2.z();
  }
}

void testSplatIo2() {
  using namespace p11;
  using namespace p11::splat;
  const torch::DeviceType device = util_torch::findBestTorchDevice();
  const int kSplatImageWidth = 4096;
  const int kSplatImageHeight = 2048;
  std::vector<SerializableSplat> splats = loadSplatPLYFile(FLAGS_src_ply);
  
  auto model = serializableSplatsToModel(device, splats);
  
  std::vector<SerializableSplat> splats2 = splatModelToSerializable(model);

  cv::Mat splat_image = encodeSplatsInImage(splats2, kSplatImageWidth, kSplatImageHeight);
  cv::imwrite(FLAGS_output_dir + "/splats.png", splat_image);
}

void testRenderFile() {
  using namespace p11;
  using namespace p11::splat;
  const torch::DeviceType device = util_torch::findBestTorchDevice();
  std::vector<SerializableSplat> splats;
  std::string ext = file::filenameExtension(FLAGS_src_ply); // HACK: you can pass a png here too!

  if (ext == "ply") splats = loadSplatPLYFile(FLAGS_src_ply);
  else if (ext == "png") splats = loadSplatImageFile(FLAGS_src_ply);
  else XCHECK(false) << "Unknown file format: " << FLAGS_src_ply;

  XCHECK(!splats.empty());

  // Create grid of splats around the camrea
  //for (int z = -10; z <= 10; ++z)
  //for (int y = -10; y <= 10; ++y)
  //for (int x = -10; x <= 10; ++x) {
  //  SerializableSplat s;
  //  s.pos = Eigen::Vector3f(float(x), float(y), float(z));
  //  s.scale = Eigen::Vector3f(-4.0, -4.0, -4.0);
  //  s.color = Eigen::Vector4f(x / 20.0 + 0.5, y / 20.0 + 0.5, z /20.0 + 0.5, 1);
  //  s.quat = Eigen::Vector4f(1, 0, 0, 0);
  //  splats.push_back(s);
  //}

  // Replace all splats with one in front and one behind the default camera pose
  //splats.clear();
  //{
  //  SerializableSplat s;
  //  s.pos = Eigen::Vector3f(0,0,5);
  //  s.scale = Eigen::Vector3f(-1.0, -1.0, -1.0);
  //  s.color = Eigen::Vector4f(1.0, 1.0, 0, 1);
  //  s.quat = Eigen::Vector4f(0, 0, 0, 1);
  //  splats.push_back(s);
  //}
  //{
  //  SerializableSplat s;
  //  s.pos = Eigen::Vector3f(0,0,-5.0);
  //  s.scale = Eigen::Vector3f(-1.0, -1.0, -1.0);
  //  s.color = Eigen::Vector4f(0.0, 1.0, 1.0, 1);
  //  s.quat = Eigen::Vector4f(0, 0, 0, 1);
  //  splats.push_back(s);
  //}


  auto model = serializableSplatsToModel(device, splats);

  for (int frame_counter = 0; frame_counter < 3000000; ++frame_counter) {
    calibration::RectilinearCamerad cam = calibration::guessRectilinearIntrinsics(512, 288, 120.0);
    //cam.focal_length *= 1.0 + 0.3 * std::sin(frame_counter * 0.01);

    if (ext == "ply") { // Camera for Splatfacto PLYs
      float theta = (3483 + frame_counter) * M_PI / 180.0;
      cam.cam_from_world.linear().row(0) = Eigen::Vector3d(-1.0, 0.0, 0.0); // right
      cam.cam_from_world.linear().row(1) = Eigen::Vector3d(0.0, 0.0, +1.0); // up
      cam.cam_from_world.linear().row(2) = Eigen::Vector3d(0.0, -1.0, 0.0); // forward
      cam.setPositionInWorld(Eigen::Vector3d(
        0.0,
        3 + 3.0 * std::sin(theta),
        0.0));
    }
    if (ext == "png") {
      float theta = 0.5 * std::sin(0.005 * frame_counter) - (M_PI / 2.0);
      const float r = 3.0 + 2.0 * std::sin(0.01 * frame_counter);
      cam.cam_from_world.linear() = Eigen::AngleAxisd(theta + M_PI/2.0, Eigen::Vector3d::UnitY()).matrix();
      cam.setPositionInWorld(r * Eigen::Vector3d(std::cos(theta), 0.0, -10 + std::sin(theta)));
    }
 
    auto [rendered_image, alpha_map, depth_map, _0, _1, _2, _3] = renderSplatImageGsplat(device, cam, model);

    auto cpu_image = rendered_image.to(torch::kCPU);
    //auto cpu_depth = depth_map.to(torch::kCPU);
    //auto cpu_alpha = alpha_map.to(torch::kCPU);
    cv::Mat cv_image(cpu_image.size(0), cpu_image.size(1), CV_32FC3, cpu_image.data_ptr<float>());
    //cv::Mat cv_depth(cpu_depth.size(0), cpu_depth.size(1), CV_32FC1, cpu_depth.data_ptr<float>());
    //cv::Mat cv_alpha(cpu_alpha.size(0), cpu_alpha.size(1), CV_32FC1, cpu_alpha.data_ptr<float>());
    cv::namedWindow("image", cv::WINDOW_NORMAL);
    cv::imshow("image", cv_image);
    //cv::imshow("depth", cv_depth);
    //cv::imshow("alpha", cv_alpha);
    cv::waitKey(1);
  }
}

int main(int argc, char** argv)
{
// Workaround a bug in libtorch for windows where it links against the wrong dll and doesn't support
// CUDA. See https://github.com/pytorch/pytorch/issues/72396
#ifdef _WIN32
  LoadLibraryA("torch_cuda.dll");
#endif

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  //testUnboundContractExpand(); return EXIT_SUCCESS;

  if (FLAGS_mode == "ply2png") {
    convertPlyToPng(); return EXIT_SUCCESS; 
  }

  if (FLAGS_mode == "iotest2") {
    testSplatIo2(); return EXIT_SUCCESS; 
  }

  if (FLAGS_mode == "render") {
    testRenderFile(); return EXIT_SUCCESS; 
  }

  XCHECK(!FLAGS_output_dir.empty());
  p11::file::createDirectoryIfNotExists(FLAGS_output_dir);
  XCHECK(p11::file::directoryExists(FLAGS_output_dir));

  p11::splat::SplatConfig cfg;
  cfg.vid_dir = FLAGS_vid_dir;
  cfg.train_images_dir = FLAGS_train_dir;
  cfg.train_json = FLAGS_train_dir + "/dataset.json";
  cfg.sfm_pointcloud = !cfg.vid_dir.empty()
    ? (cfg.vid_dir + "/sfm/track_pointcloud.bin")
    : (FLAGS_train_dir + "/pointcloud_sfm.bin");
  if (!FLAGS_sfm_pointcloud_path.empty()) cfg.sfm_pointcloud = FLAGS_sfm_pointcloud_path;
  cfg.output_dir = FLAGS_output_dir;
  cfg.save_steps = FLAGS_save_steps;
  cfg.calc_psnr = FLAGS_calc_psnr;
  cfg.resize_max_dim = FLAGS_resize_max_dim;
  cfg.max_num_splats = FLAGS_n;
  cfg.num_itrs = FLAGS_num_itrs;
  cfg.first_frame_warmup_itrs = FLAGS_first_frame_warmup_itrs;
  cfg.images_per_batch = FLAGS_images_per_batch;
  cfg.train_vis_interval = FLAGS_train_vis_interval;
  cfg.population_update_interval = FLAGS_popi;
  cfg.learning_rate = FLAGS_learning_rate;
  cfg.init_with_monodepth = FLAGS_init_with_monodepth;
  cfg.use_depth_loss = FLAGS_use_depth_loss;
  cfg.is_video = !cfg.vid_dir.empty();

  if (FLAGS_vid_dir.empty()) {
    p11::splat::runSplatPipelineStatic(cfg);
  } else {
    p11::splat::runSplatPipelineVideo(cfg);
  }
  return EXIT_SUCCESS;
}
