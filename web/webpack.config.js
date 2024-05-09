const path = require('path');

module.exports = {
  entry: './lifecast_res/LifecastVideoPlayer11.js',

  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'lifecast-video-player-bundle.js',
    library: 'LifecastVideoPlayer',
    libraryTarget: 'umd',
    globalObject: 'this',
  },

  mode: 'production',
  devtool: 'source-map',
};
