import { create } from 'zustand';
import mqtt from 'mqtt';
import { MQTT_BROKER_URL } from '../config';

/**
 * Global MQTT Store for VeinGuard.
 * Manages a persistent singleton connection and provides
 * request-response utilities for Login and User Management.
 */
export const useMqttStore = create((set, get) => ({
  client: null,
  isConnected: false,
  status: 'OFFLINE',
  clientId: `mobile-${Math.random().toString(16).slice(2, 10)}`,

  connect: () => {
    if (get().client) return;

    console.log('Connecting to MQTT:', MQTT_BROKER_URL);
    const client = mqtt.connect(MQTT_BROKER_URL, {
      clientId: get().clientId,
      clean: true,
      reconnectPeriod: 5000,
    });

    client.on('connect', () => {
      console.log('MQTT Connected');
      set({ isConnected: true, status: 'ONLINE' });
      
      // Subscribe to personal response topics
      client.subscribe(`veinguard/res/auth/login/${get().clientId}`);
      client.subscribe(`veinguard/res/users/list/${get().clientId}`);
      client.subscribe('veinguard/status');
    });

    client.on('message', (topic, payload) => {
      const message = payload.toString();
      console.log(`MQTT Received [${topic}]:`, message);

      if (topic === 'veinguard/status') {
        try {
          const data = JSON.parse(message);
          set({ status: data.status || 'ONLINE' });
        } catch (e) {}
      }
    });

    client.on('close', () => set({ isConnected: false, status: 'OFFLINE' }));
    client.on('error', (err) => console.error('MQTT Error:', err));

    set({ client });
  },

  /**
   * Helper to perform a Request-Response cycle over MQTT.
   * Useful for Login and List operations.
   */
  request: async (cmdTopic, resTopic, payload, timeout = 5000) => {
    const { client } = get();
    if (!client || !get().isConnected) {
      throw new Error('MQTT not connected');
    }

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        client.removeListener('message', handler);
        reject(new Error('MQTT Timeout'));
      }, timeout);

      const handler = (topic, msg) => {
        if (topic === resTopic) {
          clearTimeout(timer);
          client.removeListener('message', handler);
          try {
            resolve(JSON.parse(msg.toString()));
          } catch (e) {
            resolve(msg.toString());
          }
        }
      };

      client.on('message', handler);
      client.publish(cmdTopic, JSON.stringify({
        ...payload,
        client_id: get().clientId
      }));
    });
  },

  login: async (username, password) => {
    return get().request(
      'veinguard/cmd/auth/login',
      `veinguard/res/auth/login/${get().clientId}`,
      { username, password }
    );
  },

  fetchUsers: async () => {
    return get().request(
      'veinguard/cmd/users/list',
      `veinguard/res/users/list/${get().clientId}`,
      {}
    );
  },

  fetchLogs: async () => {
    return get().request(
      'veinguard/cmd/logs/list',
      `veinguard/res/logs/list/${get().clientId}`,
      {}
    );
  },

  fetchAuditLogs: async () => {
    return get().request(
      'veinguard/cmd/audit/list',
      `veinguard/res/audit/list/${get().clientId}`,
      {}
    );
  }
}));
