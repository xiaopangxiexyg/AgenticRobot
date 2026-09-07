(function questDevApp() {
  'use strict';

  const wsStatus = document.getElementById('ws-status');
  const xrStatus = document.getElementById('xr-status');
  const eventLog = document.getElementById('event-log');
  const startButton = document.getElementById('start-xr');
  const pairingTokenInput = document.getElementById('pairing-token');
  const pairingButton = document.getElementById('connect-ws');
  const pairingStatus = document.getElementById('pairing-status');

  let ws = null;
  let wsAuthenticated = false;
  let pairingToken = null;
  let reconnectTimer = null;
  let pairingRejected = false;
  let xrSession = null;
  let xrStarting = false;
  let xrEndRequested = false;
  let xrSupportedMode = null;
  let xrSupportChecked = false;
  let xrMode = 'none';
  let xrSessionId = null;
  let referenceSpaceType = null;
  let referenceSpaceGeneration = 0;
  const controllerSourceIdentity = (
    window.QuestDevControllerSourceIdentity.createTracker()
  );
  let gl = null;
  let clientConfig = {
    protocol_version: 2,
    security: { pairing_required: true },
    xr: { reference_space: 'local-floor' },
    features: {},
  };
  const clientInstanceId = randomId('browser');

  function randomId(prefix) {
    if (globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function') {
      return `${prefix}-${globalThis.crypto.randomUUID()}`;
    }
    const bytes = new Uint8Array(16);
    globalThis.crypto.getRandomValues(bytes);
    return `${prefix}-${Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('')}`;
  }

  function setStatus(element, text, ok) {
    element.textContent = text;
    element.className = ok ? 'ok' : 'bad';
  }

  function updateStartButton() {
    startButton.disabled = !(
      wsAuthenticated
      && xrSupportChecked
      && xrSupportedMode
      && !xrSession
      && !xrStarting
    );
  }

  function logEvent(value) {
    const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    eventLog.textContent = `${new Date().toLocaleTimeString()} ${text}\n${eventLog.textContent}`.slice(0, 6000);
  }

  function consumeFragmentPairingToken() {
    const params = new URLSearchParams(location.hash.replace(/^#/, ''));
    const token = params.get('pair');
    if (location.hash) history.replaceState(null, '', `${location.pathname}${location.search}`);
    return token || null;
  }

  const configReady = fetch('/api/config')
    .then((response) => {
      if (!response.ok) throw new Error(`client config HTTP ${response.status}`);
      return response.json();
    })
    .then((value) => {
      clientConfig = { ...clientConfig, ...value };
      window.QuestDevFeatures.configure(clientConfig.features, {
        logEvent,
        protocolVersion: clientConfig.protocol_version,
      });
    })
    .catch((error) => {
      logEvent(`using default client config: ${error}`);
      window.QuestDevFeatures.configure({}, {
        logEvent,
        protocolVersion: clientConfig.protocol_version,
      });
    });

  function sendRaw(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload));
  }

  function send(payload) {
    if (wsAuthenticated) sendRaw(payload);
  }

  function connectWebSocket() {
    if (!pairingToken) {
      setStatus(pairingStatus, '请输入主机终端显示的 6 位配对码', false);
      return;
    }
    clearTimeout(reconnectTimer);
    pairingRejected = false;
    if (xrSession) {
      setStatus(pairingStatus, 'ending XR before reconnecting', false);
      if (!xrEndRequested) {
        xrEndRequested = true;
        xrSession.end().catch((error) => logEvent(`XR end before reconnect failed: ${error}`));
      }
      reconnectTimer = setTimeout(connectWebSocket, 250);
      return;
    }
    if (xrStarting) {
      wsAuthenticated = false;
      updateStartButton();
      if (ws) {
        const startingSocket = ws;
        ws = null;
        startingSocket.close();
      }
      reconnectTimer = setTimeout(connectWebSocket, 250);
      return;
    }
    if (ws) {
      const previousSocket = ws;
      ws = null;
      wsAuthenticated = false;
      updateStartButton();
      previousSocket.close();
    }
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${protocol}//${location.host}/ws`);
    ws = socket;
    setStatus(wsStatus, 'connecting', false);
    socket.onopen = () => {
      if (ws !== socket) return;
      setStatus(wsStatus, 'connected; authenticating', false);
      sendRaw({
        type: 'client_auth',
        protocol_version: clientConfig.protocol_version,
        client_instance_id: clientInstanceId,
        pairing_token: pairingToken,
      });
    };
    socket.onclose = () => {
      if (ws !== socket) return;
      ws = null;
      wsAuthenticated = false;
      updateStartButton();
      setStatus(wsStatus, 'disconnected', false);
      if (xrSession && !xrEndRequested) {
        xrEndRequested = true;
        xrSession.end().catch(() => {});
      }
      if (!pairingRejected && pairingToken) {
        reconnectTimer = setTimeout(connectWebSocket, 1500);
      }
    };
    socket.onerror = () => {
      if (ws === socket) setStatus(wsStatus, 'error', false);
    };
    socket.onmessage = (event) => {
      if (ws !== socket) return;
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'client_auth_result') {
          if (message.protocol_version !== clientConfig.protocol_version) {
            pairingRejected = true;
            pairingToken = null;
            setStatus(pairingStatus, 'protocol version mismatch', false);
            socket.close();
            return;
          }
          wsAuthenticated = message.authenticated === true;
          if (wsAuthenticated) {
            pairingTokenInput.value = '';
            updateStartButton();
            setStatus(wsStatus, 'authenticated', true);
            setStatus(pairingStatus, 'paired for this page session', true);
          } else {
            pairingRejected = true;
            pairingToken = null;
            setStatus(pairingStatus, message.error || 'authentication failed', false);
            socket.close();
          }
        }
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

  configReady
    .then(() => preferredXRMode())
    .then((mode) => {
      xrSupportedMode = mode;
      xrSupportChecked = true;
      if (!mode) setStatus(xrStatus, 'immersive WebXR unavailable', false);
      updateStartButton();
    })
    .catch((error) => {
      xrSupportedMode = null;
      xrSupportChecked = true;
      setStatus(xrStatus, `WebXR support check failed: ${error}`, false);
      updateStartButton();
    });

  function sessionOptions() {
    const overlayRoot = document.getElementById('xr-overlay');
    const optionalFeatures = ['local-floor', 'bounded-floor'];
    if (!overlayRoot) return { optionalFeatures };
    return {
      optionalFeatures: [...optionalFeatures, 'dom-overlay'],
      domOverlay: { root: overlayRoot },
    };
  }

  function createXRLayer(session, context, alpha) {
    try {
      return new XRWebGLLayer(session, context, { alpha });
    } catch (error) {
      logEvent(`XRWebGLLayer alpha fallback: ${error}`);
      return new XRWebGLLayer(session, context);
    }
  }

  async function requestReferenceSpace(session, requestedType) {
    try {
      return { space: await session.requestReferenceSpace(requestedType), type: requestedType };
    } catch (error) {
      logEvent(`reference space ${requestedType} unavailable; using local: ${error}`);
      return { space: await session.requestReferenceSpace('local'), type: 'local' };
    }
  }

  function controllerSample(frame, referenceSpace, source, sourceSessionId) {
    if (!source.gripSpace) return null;
    const pose = frame.getPose(source.gripSpace, referenceSpace);
    if (!pose) return null;
    return {
      pose,
      sourceSessionId,
      handedness: source.handedness || 'none',
      targetRayMode: source.targetRayMode || 'tracked-pointer',
      profiles: Array.from(source.profiles || []),
      emulatedPosition: pose.emulatedPosition === true,
      matrixReferenceFromGrip: Array.from(pose.transform.matrix),
    };
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
        viewerPose,
        viewProjection,
        controllerPoses,
        clientInstanceId,
        xrSessionId,
        referenceSpaceType,
        referenceSpaceGeneration,
        logEvent,
      });
    }
  }

  async function startXR() {
    if (xrSession || xrStarting) return;
    if (!wsAuthenticated) throw new Error('Pair with the Quest Dev host first');
    if (!xrSupportChecked) throw new Error('WebXR support check is still pending');
    if (!xrSupportedMode) throw new Error('Quest Browser WebXR immersive-ar/vr unavailable');
    xrStarting = true;
    updateStartButton();
    let session = null;
    const startSocket = ws;
    const requireStartConnection = () => {
      if (!wsAuthenticated || !startSocket || ws !== startSocket) {
        throw new Error('authenticated WSS changed while XR was starting');
      }
    };
    try {
      const mode = xrSupportedMode;
      session = await navigator.xr.requestSession(mode, sessionOptions());
      requireStartConnection();
      const context = ensureWebGL();
      await context.makeXRCompatible();
      requireStartConnection();
      session.updateRenderState({
        baseLayer: createXRLayer(session, context, mode === 'immersive-ar'),
      });
      const requestedReferenceSpace = clientConfig.xr.reference_space || 'local-floor';
      const reference = await requestReferenceSpace(session, requestedReferenceSpace);
      requireStartConnection();
      const referenceSpace = reference.space;
      xrSession = session;
      xrMode = mode;
      xrSessionId = randomId('xr');
      referenceSpaceType = reference.type;
      referenceSpaceGeneration = 0;
      controllerSourceIdentity.reset();

      session.addEventListener('end', () => {
        if (xrSession !== session) return;
        send({
          type: 'xr_session_state',
          protocol_version: clientConfig.protocol_version,
          state: 'ended',
          client_instance_id: clientInstanceId,
          xr_session_id: xrSessionId,
          reference_space_type: referenceSpaceType,
          reference_space_generation: referenceSpaceGeneration,
        });
        window.QuestDevFeatures.onSessionEnd({ gl: context, logEvent });
        xrSession = null;
        xrEndRequested = false;
        xrMode = 'none';
        xrSessionId = null;
        referenceSpaceType = null;
        referenceSpaceGeneration = 0;
        controllerSourceIdentity.reset();
        updateStartButton();
        setStatus(xrStatus, 'ended', false);
      }, { once: true });

      window.QuestDevFeatures.onXRReady({
        gl: context,
        session,
        mode,
        referenceSpace,
        logEvent,
      });
      setStatus(xrStatus, `${mode}; ${referenceSpaceType}; generation 0`, true);
      send({
        type: 'xr_session_state',
        protocol_version: clientConfig.protocol_version,
        state: 'started',
        mode,
        client_instance_id: clientInstanceId,
        xr_session_id: xrSessionId,
        reference_space_type: referenceSpaceType,
        reference_space_generation: referenceSpaceGeneration,
      });

      referenceSpace.addEventListener('reset', () => {
        if (xrSession !== session) return;
        const previousGeneration = referenceSpaceGeneration;
        referenceSpaceGeneration += 1;
        window.QuestDevFeatures.onReferenceSpaceReset({
          referenceSpace,
          previousGeneration,
          newGeneration: referenceSpaceGeneration,
          logEvent,
        });
        send({
          type: 'reference_space_reset',
          protocol_version: clientConfig.protocol_version,
          client_instance_id: clientInstanceId,
          xr_session_id: xrSessionId,
          reference_space_type: referenceSpaceType,
          previous_generation: previousGeneration,
          new_generation: referenceSpaceGeneration,
        });
        setStatus(
          xrStatus,
          `${mode}; ${referenceSpaceType}; generation ${referenceSpaceGeneration}`,
          true,
        );
      });

      function onFrame(time, frame) {
        if (xrSession !== session) return;
        session.requestAnimationFrame(onFrame);
        const viewerPose = frame.getViewerPose(referenceSpace);
        const controllerPoses = [];
        const sourceOccurrences = new Map();
        for (const source of session.inputSources) {
          const fingerprint = controllerSourceIdentity.fingerprint(source);
          const occurrence = sourceOccurrences.get(fingerprint) || 0;
          sourceOccurrences.set(fingerprint, occurrence + 1);
          const sample = controllerSample(
            frame,
            referenceSpace,
            source,
            controllerSourceIdentity.idFor(source, occurrence),
          );
          if (sample) controllerPoses.push(sample);
        }
        window.QuestDevFeatures.onXRFrame({
          time,
          frame,
          viewerPose,
          controllerPoses,
          clientInstanceId,
          xrSessionId,
          referenceSpaceType,
          referenceSpaceGeneration,
          protocolVersion: clientConfig.protocol_version,
          send,
          logEvent,
        });
        renderFrame(session, viewerPose, controllerPoses);
      }

      session.requestAnimationFrame(onFrame);
    } catch (error) {
      if (session) await session.end().catch(() => {});
      if (xrSession === session) {
        window.QuestDevFeatures.onSessionEnd({ gl, logEvent });
        xrSession = null;
        xrEndRequested = false;
        xrMode = 'none';
        xrSessionId = null;
        referenceSpaceType = null;
        referenceSpaceGeneration = 0;
        controllerSourceIdentity.reset();
      }
      throw error;
    } finally {
      xrStarting = false;
      updateStartButton();
    }
  }

  pairingButton.addEventListener('click', () => {
    pairingToken = pairingTokenInput.value.trim() || null;
    connectWebSocket();
  });
  startButton.addEventListener('click', () => startXR().catch((error) => {
    setStatus(xrStatus, String(error), false);
    logEvent(String(error));
  }));
  window.addEventListener('error', (event) => logEvent(event.message));
  window.addEventListener('unhandledrejection', (event) => logEvent(String(event.reason)));

  setStatus(wsStatus, 'disconnected', false);
  setStatus(xrStatus, 'not started', false);
  startButton.disabled = true;
  configReady.then(() => {
    pairingToken = consumeFragmentPairingToken();
    if (pairingToken) {
      connectWebSocket();
    } else {
      setStatus(pairingStatus, '请输入主机终端显示的 6 位配对码', false);
    }
  });
}());
