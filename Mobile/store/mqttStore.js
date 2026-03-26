import { create } from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import mqtt from 'mqtt';
import {
  deleteUserProfile,
  syncAccessEvent,
  syncBiometricProfile,
  syncTelemetry,
  syncUserProfile,
} from '../services/cloudSync';
import {
  buildMqttBrokerUrl,
  MQTT_DEFAULT_HOST,
  MQTT_DEFAULT_PASSWORD,
  MQTT_DEFAULT_PORT,
  MQTT_DEFAULT_USERNAME,
  MQTT_DEFAULT_WS_PORT,
  MQTT_TOPICS,
  responseTopic,
} from '../config';

const MQTT_CONFIG_KEY = 'mqtt_connection_config';

const getDefaultBrokerConfig = () => ({
  host: MQTT_DEFAULT_HOST,
  wsPort: MQTT_DEFAULT_WS_PORT,
  mqttPort: MQTT_DEFAULT_PORT,
  username: MQTT_DEFAULT_USERNAME,
  password: MQTT_DEFAULT_PASSWORD,
});

const sanitizeHost = (value) =>
  String(value || '')
    .trim()
    .replace(/^wss?:\/\//, '')
    .replace(/\/+$/, '');

const normalizeBrokerConfig = (value = {}) => ({
  host: sanitizeHost(value.host) || MQTT_DEFAULT_HOST,
  wsPort: String(value.wsPort || MQTT_DEFAULT_WS_PORT).trim() || MQTT_DEFAULT_WS_PORT,
  mqttPort: String(value.mqttPort || MQTT_DEFAULT_PORT).trim() || MQTT_DEFAULT_PORT,
  username: String(value.username ?? MQTT_DEFAULT_USERNAME).trim() || MQTT_DEFAULT_USERNAME,
  password: String(value.password ?? MQTT_DEFAULT_PASSWORD),
});

export const useMqttStore = create((set, get) => ({
  client: null,
  isConnected: false,
  status: 'OFFLINE',
  statusPayload: null,
  telemetry: null,
  settingsAck: null,
  lastScanResult: null,
  lastError: null,
  configReady: false,
  brokerConfig: getDefaultBrokerConfig(),
  clientId: `mobile-${Math.random().toString(16).slice(2, 10)}`,

  bootstrap: async () => {
    try {
      const raw = await AsyncStorage.getItem(MQTT_CONFIG_KEY);
      const saved = raw ? JSON.parse(raw) : {};
      set({
        brokerConfig: normalizeBrokerConfig(saved),
        configReady: true,
      });
    } catch {
      set({
        brokerConfig: getDefaultBrokerConfig(),
        configReady: true,
      });
    }
  },

  disconnect: () => {
    const client = get().client;
    if (client) {
      try {
        client.end(true);
      } catch {}
    }
    set({ client: null, isConnected: false, status: 'OFFLINE' });
  },

  updateBrokerConfig: async (partialConfig, reconnect = true) => {
    const nextConfig = normalizeBrokerConfig({
      ...get().brokerConfig,
      ...partialConfig,
    });

    await AsyncStorage.setItem(MQTT_CONFIG_KEY, JSON.stringify(nextConfig));
    set({ brokerConfig: nextConfig });

    if (reconnect) {
      get().disconnect();
      await get().connect();
    }

    return nextConfig;
  },

  connect: async () => {
    if (!get().configReady) {
      await get().bootstrap();
    }

    if (get().client) return;

    const brokerConfig = normalizeBrokerConfig(get().brokerConfig);
    const brokerUrl = buildMqttBrokerUrl(brokerConfig);
    let client = null;

    try {
      client = mqtt.connect(brokerUrl, {
        clientId: get().clientId,
        clean: true,
        reconnectPeriod: 5000,
        username: brokerConfig.username || undefined,
        password: brokerConfig.password || undefined,
      });
    } catch (error) {
      set({
        client: null,
        isConnected: false,
        status: 'ERROR',
        lastError: error?.message || 'MQTT connection error',
      });
      return;
    }

    client.on('connect', () => {
      try {
        set({ isConnected: true, status: 'ONLINE', lastError: null });
        client.subscribe(responseTopic('auth/login', get().clientId));
        client.subscribe(responseTopic('users/list', get().clientId));
        client.subscribe(responseTopic('users/update', get().clientId));
        client.subscribe(responseTopic('users/delete', get().clientId));
        client.subscribe(responseTopic('access/logs', get().clientId));
        client.subscribe(responseTopic('audit/list', get().clientId));
        client.subscribe(responseTopic('access/scan', get().clientId));
        client.subscribe(responseTopic('users/enroll', get().clientId));
        client.subscribe(responseTopic('settings/update', get().clientId));
        client.subscribe(MQTT_TOPICS.status);
        client.subscribe(MQTT_TOPICS.telemetry);
      } catch (error) {
        set({ lastError: error?.message || 'MQTT subscribe error' });
      }
    });

    client.on('message', (topicName, payload) => {
      try {
        const data = JSON.parse(payload.toString());
        if (topicName === MQTT_TOPICS.status) {
          set({ status: data.status || 'ONLINE', statusPayload: data });
          return;
        }

        if (topicName === MQTT_TOPICS.telemetry) {
          set({ telemetry: data });
          const sanitizedCamera = data.camera
            ? Object.fromEntries(
                Object.entries(data.camera).filter(([key]) => key !== 'preview_jpeg_base64'),
              )
            : undefined;
          syncTelemetry(data.device_id || 'rpi-entry-01', {
            ...data,
            ...(sanitizedCamera ? { camera: sanitizedCamera } : {}),
          }).catch(() => {});
          return;
        }

        if (topicName === responseTopic('settings/update', get().clientId)) {
          set({ settingsAck: data, telemetry: data.telemetry || get().telemetry });
        }
      } catch (error) {
        if (topicName === MQTT_TOPICS.status) {
          set({ status: 'ONLINE' });
          return;
        }
        set({ lastError: error?.message || 'MQTT message error' });
      }
    });

    client.on('close', () => {
      set({ isConnected: false, status: 'OFFLINE', statusPayload: null, client: null });
    });
    client.on('error', (error) => {
      set({ isConnected: false, status: 'ERROR', lastError: error?.message || 'MQTT error' });
    });

    set({ client });
  },

  request: async (cmdTopic, resTopic, payload, timeout = 7000) => {
    let { client } = get();
    if (!client || !get().isConnected) {
      get().disconnect();
      await get().connect();

      const startedAt = Date.now();
      while (!get().isConnected && Date.now() - startedAt < 8000) {
        await new Promise((resolve) => setTimeout(resolve, 200));
      }

      if (!get().isConnected) {
        throw new Error(get().lastError || 'MQTT not connected');
      }

      client = get().client;
    }

    if (!client) {
      throw new Error('MQTT client unavailable');
    }

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        client.removeListener('message', handler);
        reject(new Error('MQTT timeout'));
      }, timeout);

      const handler = (topicName, msg) => {
        if (topicName !== resTopic) return;
        clearTimeout(timer);
        client.removeListener('message', handler);
        try {
          resolve(JSON.parse(msg.toString()));
        } catch {
          resolve(msg.toString());
        }
      };

      client.on('message', handler);
      try {
        client.publish(
          cmdTopic,
          JSON.stringify({
            ...payload,
            client_id: get().clientId,
          }),
          (error) => {
            if (!error) {
              return;
            }
            clearTimeout(timer);
            client.removeListener('message', handler);
            reject(error);
          },
        );
      } catch (error) {
        clearTimeout(timer);
        client.removeListener('message', handler);
        reject(error);
      }
    });
  },

  login: async (username, password) =>
    get().request(
      MQTT_TOPICS.loginCmd,
      responseTopic('auth/login', get().clientId),
      { username, password }
    ),

  fetchUsers: async () =>
    get().request(
      MQTT_TOPICS.usersCmd,
      responseTopic('users/list', get().clientId),
      {}
    ),

  fetchLogs: async () =>
    get().request(
      MQTT_TOPICS.logsCmd,
      responseTopic('access/logs', get().clientId),
      {}
    ),

  fetchAuditLogs: async () =>
    get().request(
      MQTT_TOPICS.auditCmd,
      responseTopic('audit/list', get().clientId),
      {}
    ),

  enrollUser: async (payload) =>
    get().request(
      MQTT_TOPICS.enrollCmd,
      responseTopic('users/enroll', get().clientId),
      payload,
      30000
    ).then(async (response) => {
      if (response?.status === 'success') {
        await syncUserProfile(response.user_id, {
          user_id: response.user_id,
          username: response.username,
          email: payload.email,
          department: payload.department,
          role: payload.role,
          biometric_key: response.biometric_key,
          sample_count: response.sample_count,
        }).catch(() => {});
        await syncBiometricProfile(response.user_id, {
          user_id: response.user_id,
          biometric_key: response.biometric_key,
          profile: response.profile,
          sample_count: response.sample_count,
        }).catch(() => {});
      }
      return response;
    }),

  updateUser: async (payload) =>
    get().request(
      MQTT_TOPICS.usersUpdateCmd,
      responseTopic('users/update', get().clientId),
      payload
    ).then(async (response) => {
      if (response?.status === 'success' && response.user) {
        await syncUserProfile(response.user.id, response.user).catch(() => {});
      }
      return response;
    }),

  deleteUser: async (userId) =>
    get().request(
      MQTT_TOPICS.usersDeleteCmd,
      responseTopic('users/delete', get().clientId),
      { user_id: userId }
    ).then(async (response) => {
      if (response?.status === 'success') {
        await deleteUserProfile(userId).catch(() => {});
      }
      return response;
    }),

  triggerScan: async (userId) =>
    get().request(
      MQTT_TOPICS.scanCmd,
      responseTopic('access/scan', get().clientId),
      userId ? { user_id: userId } : {},
      12000
    ).then(async (response) => {
      if (response?.event?.id) {
        set({ lastScanResult: response });
        await syncAccessEvent(response.event.id, response.event).catch(() => {});
      }
      return response;
    }),

  updateSystemSettings: async (payload) =>
    get().request(
      MQTT_TOPICS.settingsCmd,
      responseTopic('settings/update', get().clientId),
      payload
    ),
}));
