(function questDevApp() {
  'use strict';

  const wsStatus = document.getElementById('ws-status');
  const xrStatus = document.getElementById('xr-status');
  const eventLog = document.getElementById('event-log');
  const startButton = document.getElementById('start-xr');

  let ws = null;
  let reconnectTimer = null;
  let xrSession = null;
  let xrMode = 'none';
  let gl = null;
  let clientConfig = {
    protocol_version: 1,
    xr: { reference_space: 'local-floor' },
    features: {},
  };

  function setStatus(element, text, ok) {
    element.textContent = text;
    element.className = ok ? 'ok' : 'bad';
  }

  function logEvent(value) {
    const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    eventLog.textContent = `${new Date().toLocaleTimeString()} ${text}\n${eventLog.textContent}`.slice(0, 6000);
  }

  const configReady = fetch('/api/config')
    .then((response) => {
      if (!response.ok) throw new Error(`client config HTTP ${response.status}`);
      return response.json();
    })
    .then((value) => {
      clientConfig = { ...clientConfig, ...value };
      window.QuestDevFeatures.configure(clientConfig.features, { logEvent });
    })
    .catch((error) => {
      logEvent(`using default client config: ${error}`);
      window.QuestDevFeatures.configure({}, { logEvent });
    });

  function send(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload));
  }

  function connectWebSocket() {
    clearTimeout(reconnectTimer);
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);
    setStatus(wsStatus, 'connecting', false);
    ws.onopen = () => {
      setStatus(wsStatus, 'connected', true);
      send({ type: 'client_hello', client: 'quest_dev_web', protocol_version: 1 });
    };
    ws.onclose = () => {
      setStatus(wsStatus, 'disconnected', false);
      reconnectTimer = setTimeout(connectWebSocket, 1500);
    };
    ws.onerror = () => setStatus(wsStatus, 'error', false);
    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        window.QuestDevFeatures.handleServerMessage(message, { logEvent });
        logEvent(message);
      } catch (error) {
        logEvent(`invalid server message: ${error}`);
      }
    };
  }

  function ensureWebGL() {
    if (gl) return gl;
    const canvas = document.createElement('canvas');
    canvas.hidden = true;
    document.body.appendChild(canvas);
    gl = canvas.getContext('webgl', { xrCompatible: true, antialias: true, alpha: true });
    if (!gl) throw new Error('WebGL unavailable');
    return gl;
  }

  async function preferredXRMode() {
    if (!navigator.xr) return null;
    if (await navigator.xr.isSessionSupported('immersive-ar').catch(() => false)) return 'immersive-ar';
    if (await navigator.xr.isSessionSupported('immersive-vr').catch(() => false)) return 'immersive-vr';
    return null;
  }

  function sessionOptions() {
    return { optionalFeatures: ['local-floor', 'bounded-floor'] };
  }

  function createXRLayer(session, context, alpha) {
    try {
      return new XRWebGLLayer(session, context, { alpha });
    } catch (error) {
      logEvent(`XRWebGLLayer alpha fallback: ${error}`);
      return new XRWebGLLayer(session, context);
    }
  }

  function renderFrame(session, viewerPose, controllerPoses) {
    const layer = session.renderState.baseLayer;
    if (!layer || !viewerPose) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0, 0, 0, xrMode === 'immersive-ar' ? 0 : 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);

    for (const view of viewerPose.views) {
      const viewport = layer.getViewport(view);
      gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
      const viewProjection = window.QuestDevXRMath.mat4Multiply(
        view.projectionMatrix,
        view.transform.inverse.matrix,
      );
      window.QuestDevFeatures.renderView({
        gl,
        session,
        view,
        viewProjection,
        controllerPoses,
        logEvent,
      });
    }
  }

  async function startXR() {
    if (xrSession) return;
    await configReady;
    const mode = await preferredXRMode();
    if (!mode) throw new Error('Quest Browser WebXR immersive-ar/vr unavailable');
    const context = ensureWebGL();
    await context.makeXRCompatible();
    const session = await navigator.xr.requestSession(mode, sessionOptions());
    session.updateRenderState({ baseLayer: createXRLayer(session, context, mode === 'immersive-ar') });
    const referenceSpaceName = clientConfig.xr.reference_space || 'local-floor';
    const referenceSpace = await session.requestReferenceSpace(referenceSpaceName)
      .catch(() => session.requestReferenceSpace('local'));
    xrSession = session;
    xrMode = mode;
    window.QuestDevFeatures.onXRReady({ gl: context, session, mode, referenceSpace, logEvent });
    setStatus(xrStatus, mode, true);
    send({ type: 'xr_session_state', state: 'started', mode });

    referenceSpace.addEventListener('reset', () => {
      window.QuestDevFeatures.onReferenceSpaceReset({ referenceSpace, logEvent });
      send({ type: 'reference_space_reset', timestamp: Date.now() / 1000 });
    });

    function onFrame(time, frame) {
      void time;
      session.requestAnimationFrame(onFrame);
      const viewerPose = frame.getViewerPose(referenceSpace);
      const controllerPoses = [];
      for (const source of session.inputSources) {
        if (!source.gripSpace) continue;
        const pose = frame.getPose(source.gripSpace, referenceSpace);
        if (pose) controllerPoses.push(pose);
      }
      renderFrame(session, viewerPose, controllerPoses);
    }

    session.requestAnimationFrame(onFrame);
    session.addEventListener('end', () => {
      window.QuestDevFeatures.onSessionEnd({ logEvent });
      xrSession = null;
      xrMode = 'none';
      setStatus(xrStatus, 'ended', false);
      send({ type: 'xr_session_state', state: 'ended' });
    });
  }

  startButton.addEventListener('click', () => startXR().catch((error) => {
    setStatus(xrStatus, String(error), false);
    logEvent(String(error));
  }));
  window.addEventListener('error', (event) => logEvent(event.message));
  window.addEventListener('unhandledrejection', (event) => logEvent(String(event.reason)));

  setStatus(wsStatus, 'disconnected', false);
  setStatus(xrStatus, 'not started', false);
  connectWebSocket();
}());
