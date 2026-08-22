// Extracted from quest_pose/web/app.js. Matrices follow WebXR/OpenGL
// column-major layout.
(function exposeQuestDevXRMath(global) {
  'use strict';

  function mat4Multiply(a, b) {
    const out = new Float32Array(16);
    for (let col = 0; col < 4; col += 1) {
      for (let row = 0; row < 4; row += 1) {
        out[col * 4 + row] =
          a[row] * b[col * 4] +
          a[4 + row] * b[col * 4 + 1] +
          a[8 + row] * b[col * 4 + 2] +
          a[12 + row] * b[col * 4 + 3];
      }
    }
    return out;
  }

  function mat4Identity() {
    return new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
  }

  function posePosition(matrix) {
    return [matrix[12], matrix[13], matrix[14]];
  }

  global.QuestDevXRMath = { mat4Multiply, mat4Identity, posePosition };
}(window));
