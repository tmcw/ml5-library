// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
PoseNet
*/

import * as tf from '@tensorflow/tfjs';
import * as posenet from '@tensorflow-models/posenet';
import EventEmitter from 'events';

const DEFAULTS = {
  imageScaleFactor: 0.3,
  outputStride: 16,
  flipHorizontal: false,
  minConfidence: 0.5,
  maxPoseDetections: 5,
  scoreThreshold: 0.5,
  nmsRadius: 20,
  detectionType: 'multiple',
  multiplier: 0.75,
};

class PoseNet extends EventEmitter {
  constructor(video, options, detectionType, callback = () => {}) {
    super();
    this.video = video;
    this.detectionType = detectionType || DEFAULTS.detectionType;
    this.imageScaleFactor = options.imageScaleFactor || DEFAULTS.imageScaleFactor;
    this.outputStride = options.outputStride || DEFAULTS.outputStride;
    this.flipHorizontal = options.flipHorizontal || DEFAULTS.flipHorizontal;
    this.minConfidence = options.minConfidence || DEFAULTS.minConfidence;
    this.multiplier = options.multiplier || DEFAULTS.multiplier;
    this.ready = this.load().then(() => {
      callback();
      return this;
    });
  }

  async load() {
    const net = await posenet.load(this.multiplier);
    this.net = net;
    if (this.video) {
      await new Promise((resolve) => {
        this.video.onplay = resolve;
      });
      if (this.detectionType === 'single') {
        return this.singlePose();
      }
      return this.multiPose();
    }
    return this;
  }

  skeleton(keypoints, confidence = this.minConfidence) {
    return posenet.getAdjacentKeyPoints(keypoints, confidence);
  }

  /* eslint max-len: ["error", { "code": 180 }] */
  async singlePose(maybeInput) {
    let input;
    if (maybeInput instanceof HTMLImageElement || maybeInput instanceof HTMLVideoElement) {
      input = maybeInput;
    } else if (typeof maybeInput === 'object' && (maybeInput.elt instanceof HTMLImageElement || maybeInput.elt instanceof HTMLVideoElement)) {
      input = maybeInput.elt; // Handle p5.js image and video
    } else if (typeof maybeInput === 'function' && this.video) {
      input = this.video;
    }

    const pose = await this.net.estimateSinglePose(input, this.imageScaleFactor, this.flipHorizontal, this.outputStride);
    this.emit('pose', [{ pose, skeleton: this.skeleton(pose.keypoints) }]);
    return tf.nextFrame().then(() => this.singlePose());
  }

  async multiPose(maybeInput) {
    let input;
    if (maybeInput instanceof HTMLImageElement || maybeInput instanceof HTMLVideoElement) {
      input = maybeInput;
    } else if (typeof maybeInput === 'object' && (maybeInput.elt instanceof HTMLImageElement || maybeInput.elt instanceof HTMLVideoElement)) {
      input = maybeInput.elt; // Handle p5.js image and video
    } else if (!input && this.video) {
      input = this.video;
    }

    const poses = await this.net.estimateMultiplePoses(input, this.imageScaleFactor, this.flipHorizontal, this.outputStride);
    const result = poses.map(pose => ({ pose, skeleton: this.skeleton(pose.keypoints) }));
    this.emit('pose', result);
    return tf.nextFrame().then(() => this.multiPose());
  }
}

const poseNet = (videoOrOptionsOrCallback, optionsOrCallback, cb = () => {}) => {
  let video;
  let options = {};
  let callback = cb;
  let detectionType = null;

  if (videoOrOptionsOrCallback instanceof HTMLVideoElement) {
    video = videoOrOptionsOrCallback;
  } else if (typeof videoOrOptionsOrCallback === 'object' && videoOrOptionsOrCallback.elt instanceof HTMLVideoElement) {
    video = videoOrOptionsOrCallback.elt; // Handle a p5.js video element
  } else if (typeof videoOrOptionsOrCallback === 'object') {
    options = videoOrOptionsOrCallback;
  } else if (typeof videoOrOptionsOrCallback === 'function') {
    callback = videoOrOptionsOrCallback;
  }

  if (typeof optionsOrCallback === 'object') {
    options = optionsOrCallback;
  } else if (typeof optionsOrCallback === 'function') {
    callback = optionsOrCallback;
  } else if (typeof optionsOrCallback === 'string') {
    detectionType = optionsOrCallback;
  }

  const instance = new PoseNet(video, options, detectionType, callback);
  return callback ? instance : instance.ready;
};

export default poseNet;
