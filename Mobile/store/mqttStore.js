import { create } from 'zustand';
import mqtt from 'mqtt';
import { MQTT_BROKER_URL, MQTT_PASSWORD, MQTT_TOPICS, MQTT_USERNAME, responseTopic } from '../config';

export const useMqttStore = create((set, get) => ({
  client: null,
  isConnected: false,
  status: 'OFFLINE',
  telemetry: null,
  settingsAck: null,
  clientId: `mobile-${Math.random().toString(16).slice(2, 10)}`,

  connect: () => {
    if (get().client) return;

    const client = mqtt.connect(MQTT_BROKER_URL, {
      clientId: get().clientId,
      clean: true,
      reconnectPeriod: 5000,
      username: MQTT_USERNAME,
      password: MQTT_PASSWORD,
    });

    client.on('connect', () => {
      set({ isConnected: true, status: 'ONLINE' });
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
    });

    client.on('message', (topicName, payload) => {
      try {
        const data = JSON.parse(payload.toString());
        if (topicName === MQTT_TOPICS.status) {
          set({ status: data.status || 'ONLINE' });
          return;
        }

        if (topicName === MQTT_TOPICS.telemetry) {
          set({ telemetry: data });
          return;
        }

        if (topicName === responseTopic('settings/update', get().clientId)) {
          set({ settingsAck: data, telemetry: data.telemetry || get().telemetry });
        }
      } catch {
        if (topicName === MQTT_TOPICS.status) {
          set({ status: 'ONLINE' });
        }
      }
    });

    client.on('close', () => set({ isConnected: false, status: 'OFFLINE', client: null }));
    client.on('error', () => set({ isConnected: false, status: 'ERROR' }));

    set({ client });
  },

  request: async (cmdTopic, resTopic, payload, timeout = 7000) => {
    const { client } = get();
    if (!client || !get().isConnected) {
      throw new Error('MQTT not connected');
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
      client.publish(
        cmdTopic,
        JSON.stringify({
          ...payload,
          client_id: get().clientId,
        })
      );
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

  updateUser: async (payload) =>
    get().request(
      MQTT_TOPICS.usersUpdateCmd,
      responseTopic('users/update', get().clientId),
      payload
    ),

  deleteUser: async (userId) =>
    get().request(
      MQTT_TOPICS.usersDeleteCmd,
      responseTopic('users/delete', get().clientId),
      { user_id: userId }
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

  triggerScan: async (userId) =>
    get().request(
      MQTT_TOPICS.scanCmd,
      responseTopic('access/scan', get().clientId),
      { user_id: userId }
    ),

  enrollUser: async (payload) =>
    get().request(
      MQTT_TOPICS.enrollCmd,
      responseTopic('users/enroll', get().clientId),
      payload
    ),

  updateSystemSettings: async (payload) =>
    get().request(
      MQTT_TOPICS.settingsCmd,
      responseTopic('settings/update', get().clientId),
      payload
    ),
}));
