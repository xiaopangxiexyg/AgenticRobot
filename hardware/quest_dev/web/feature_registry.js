(function exposeQuestDevFeatureRegistry(global) {
  'use strict';

  const features = new Map();

  function register(name, feature) {
    if (!name || features.has(name)) throw new Error(`duplicate Quest Dev feature: ${name}`);
    features.set(name, { enabled: false, implementation: feature });
  }

  function configure(config, context) {
    for (const [name, entry] of features) {
      const featureConfig = (config && config[name]) || { enabled: false };
      entry.enabled = featureConfig.enabled === true;
      if (entry.implementation.configure) {
        entry.implementation.configure(featureConfig, { ...context, enabled: entry.enabled });
      }
    }
  }

  function invoke(hook, context) {
    for (const entry of features.values()) {
      if (!entry.enabled) continue;
      const callback = entry.implementation[hook];
      if (typeof callback === 'function') callback(context);
    }
  }

  function handleServerMessage(message, context) {
    invoke('handleServerMessage', { ...context, message });
  }

  global.QuestDevFeatures = {
    register,
    configure,
    handleServerMessage,
    onXRReady: (context) => invoke('onXRReady', context),
    onXRFrame: (context) => invoke('onXRFrame', context),
    renderView: (context) => invoke('renderView', context),
    onReferenceSpaceReset: (context) => invoke('onReferenceSpaceReset', context),
    onSessionEnd: (context) => invoke('onSessionEnd', context),
  };
}(window));
