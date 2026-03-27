export const APP_NAME = 'BioGuard Access';

const readBooleanEnv = (value, defaultValue = false) => {
  if (value == null || value === '') {
    return defaultValue;
  }

  return ['1', 'true', 'yes', 'on'].includes(String(value).trim().toLowerCase());
};

export const MQTT_DEFAULT_HOST = '172.16.9.115';
export const MQTT_DEFAULT_WS_PORT = '9090';
export const MQTT_DEFAULT_PORT = '1883';
export const MQTT_DEFAULT_USERNAME = 'admin';
export const MQTT_DEFAULT_PASSWORD = 'admin1234';
export const MQTT_TOPIC_PREFIX = 'bioguard';

export const buildMqttBrokerUrl = ({ host, wsPort }) => `ws://${host}:${wsPort}`;

export const topic = (path) => `${MQTT_TOPIC_PREFIX}/${path.replace(/^\/+/, '')}`;
export const responseTopic = (command, clientId) =>
  topic(`res/${command.replace(/^\/+/, '')}/${clientId}`);

export const MQTT_TOPICS = {
  status: topic('status'),
  telemetry: topic('telemetry'),
  loginCmd: topic('cmd/auth/login'),
  usersCmd: topic('cmd/users/list'),
  enrollCmd: topic('cmd/users/enroll'),
  usersUpdateCmd: topic('cmd/users/update'),
  usersDeleteCmd: topic('cmd/users/delete'),
  scanCmd: topic('cmd/access/scan'),
  previewCmd: topic('cmd/camera/preview'),
  logsCmd: topic('cmd/access/logs'),
  auditCmd: topic('cmd/audit/list'),
  settingsCmd: topic('cmd/settings/update'),
};

export const FIREBASE_CONFIG = {
  apiKey: process.env.EXPO_PUBLIC_FIREBASE_API_KEY || 'AIzaSyBKSvCUfQXY6xRMAQuW5KLNEgj3WSaSBpA',
  authDomain: process.env.EXPO_PUBLIC_FIREBASE_AUTH_DOMAIN || 'veinguard-d127f.firebaseapp.com',
  projectId: process.env.EXPO_PUBLIC_FIREBASE_PROJECT_ID || 'veinguard-d127f',
  storageBucket: process.env.EXPO_PUBLIC_FIREBASE_STORAGE_BUCKET || 'veinguard-d127f.firebasestorage.app',
  messagingSenderId: process.env.EXPO_PUBLIC_FIREBASE_MESSAGING_SENDER_ID || '375240610666',
  appId: process.env.EXPO_PUBLIC_FIREBASE_APP_ID || '1:375240610666:web:860ae203642a2ccd654b13',
  measurementId: process.env.EXPO_PUBLIC_FIREBASE_MEASUREMENT_ID || 'G-DCZ2V8JV0B',
};

export const FIREBASE_ENABLED = readBooleanEnv(process.env.EXPO_PUBLIC_FIREBASE_AUTH_ENABLED, true);
export const FIRESTORE_ENABLED = readBooleanEnv(process.env.EXPO_PUBLIC_FIRESTORE_ENABLED, false);
