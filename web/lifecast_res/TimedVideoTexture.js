/*
The MIT License

Copyright © 2010-2021 three.js authors
Copyright © 2021 Lifecast Incorporated

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

import * as THREE from './three149.module.min.js';

class TimedVideoTexture extends THREE.Texture {
  // Note: I moved the arguments around to put format and type up front.
  constructor( video, format, type, frame_callback, framerate, mapping, wrapS, wrapT, magFilter, minFilter, anisotropy ) {
    super( video, mapping, wrapS, wrapT, magFilter, minFilter, format, type, anisotropy );
    this.video = video;
    this.format = format !== undefined ? format : THREE.RGBAFormat;
    this.minFilter = minFilter !== undefined ? minFilter : THREE.LinearFilter;
    this.magFilter = magFilter !== undefined ? magFilter : THREE.LinearFilter;
    this.generateMipmaps = false;
    if (framerate == undefined) framerate = 30;

    const scope = this;

    function updateVideo(now, metadata) {
      const frame_index = Math.floor(0.5 + metadata.mediaTime * framerate);
      frame_callback(frame_index);

      scope.needsUpdate = true;
      video.requestVideoFrameCallback( updateVideo );
    }

    // For now this seems to only be supported in Chrome... bummer.
    if ('requestVideoFrameCallback' in video) {
      video.requestVideoFrameCallback( updateVideo );
    } else {
      console.log("requestVideoFrameCallback NOT SUPPORTED");
    }
  }

  clone() {
    return new this.constructor( this.image ).copy( this );
  }

  update() {
    const video = this.image;
    const hasVideoFrameCallback = 'requestVideoFrameCallback' in video;
    if ( hasVideoFrameCallback === false && video.readyState >= video.HAVE_CURRENT_DATA ) {
      this.needsUpdate = true;
    }
  }
}

TimedVideoTexture.prototype.isVideoTexture = true;

export { TimedVideoTexture };
