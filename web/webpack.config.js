const path = require('path');

module.exports = {
  entry: './lifecast_res/LifecastVideoPlayer11.js',

  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'lifecast.min.js',
    library: 'LifecastVideoPlayer',
    libraryTarget: 'umd',
    globalObject: 'this',
  },

  mode: 'production',
};
