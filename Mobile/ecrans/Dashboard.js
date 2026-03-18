import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  StatusBar,
  Animated,
} from 'react-native';
import { useTranslation } from 'react-i18next';

const COLORS = {
  bg: '#080e1a',
  cardBg: '#0d1b2e',
  cardBorder: '#1a3a5c',
  green: '#00ff88',
  greenDark: '#002a1a',
  teal: '#00e5ff',
  amber: '#e6a020',
  red: '#ff3d5a',
  redDark: '#2a0010',
  purple: '#c000ff',
  text: '#b8cfe0',
  textDim: '#4a6a8a',
  white: '#ffffff',
  headerBg: '#0a1525',
};

const devices = [
  {
    id: 'ESP32-01',
    role: 'PRIMARY SCANNER',
    status: 'ONLINE',
    statusColor: '#00ff88',
    borderColor: '#00ff88',
    heartbeat: '2s ago',
    rssi: '-42 dBm',
    battery: '87%',
    batteryColor: '#00ff88',
    firmware: 'v2.4.1',
    dot: '#00ff88',
  },
  {
    id: 'ESP32-02',
    role: 'SECONDARY SCANNER',
    status: 'OFFLINE',
    statusColor: '#ff3d5a',
    borderColor: '#ff3d5a',
    heartbeat: '5m ago',
    rssi: '-',
    battery: '12%',
    batteryColor: '#ff3d5a',
    firmware: 'v2.3.8',
    dot: '#ff3d5a',
  },
  {
    id: 'ESP32-03',
    role: 'ACCESS CONTROL',
    status: 'ONLINE',
    statusColor: '#00ff88',
    borderColor: '#00ff88',
    heartbeat: '1s ago',
    rssi: '-38 dBm',
    battery: 'AC Mains',
    batteryColor: '#e6a020',
    firmware: 'v2.4.1',
    dot: '#00ff88',
  },
];

function PulsingDot({ color }) {
  const anim = useRef(new Animated.Value(1)).current;
  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(anim, { toValue: 0.3, duration: 800, useNativeDriver: true }),
        Animated.timing(anim, { toValue: 1, duration: 800, useNativeDriver: true }),
      ])
    ).start();
  }, []);
  return (
    <Animated.View
      style={[styles.statusDot, { backgroundColor: color, opacity: anim }]}
    />
  );
}

function DeviceCard({ device }) {
  return (
    <View style={[styles.deviceCard, { borderColor: device.borderColor + '55' }]}>
      {/* Card header */}
      <View style={styles.deviceHeader}>
        <View>
          <Text style={[styles.deviceId, { color: device.statusColor === '#00ff88' ? COLORS.teal : COLORS.red }]}>
            {device.id}
          </Text>
          <Text style={styles.deviceRole}>{device.role}</Text>
        </View>
        <PulsingDot color={device.dot} />
      </View>

      {/* Stats */}
      <View style={styles.deviceStats}>
        <View style={styles.deviceRow}>
          <Text style={styles.deviceLabel}>Status</Text>
          <Text style={[styles.deviceValue, { color: device.statusColor }]}>{device.status}</Text>
        </View>
        <View style={styles.deviceRow}>
          <Text style={styles.deviceLabel}>Heartbeat</Text>
          <Text style={styles.deviceValue}>{device.heartbeat}</Text>
        </View>
        <View style={styles.deviceRow}>
          <Text style={styles.deviceLabel}>RSSI</Text>
          <Text style={styles.deviceValue}>{device.rssi}</Text>
        </View>
        <View style={styles.deviceRow}>
          <Text style={styles.deviceLabel}>Battery</Text>
          <Text style={[styles.deviceValue, { color: device.batteryColor }]}>{device.battery}</Text>
        </View>
        <View style={[styles.deviceRow, { borderBottomWidth: 0 }]}>
          <Text style={styles.deviceLabel}>Firmware</Text>
          <Text style={styles.deviceValue}>{device.firmware}</Text>
        </View>
      </View>
    </View>
  );
}

export default function Dashboard({ navigation }) {
  const { t } = useTranslation();
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (d) => {
    const h = String(d.getHours()).padStart(2, '0');
    const m = String(d.getMinutes()).padStart(2, '0');
    const s = String(d.getSeconds()).padStart(2, '0');
    return `${h}:${m}:${s}`;
  };

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.headerBg} />

      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Text style={styles.logoVein}>VEIN</Text>
          <Text style={styles.logoGuard}>GUARD</Text>
          <View style={styles.mqttBadge}>
            <PulsingDot color={COLORS.green} />
            <Text style={styles.mqttText}>MQTT ONLINE</Text>
          </View>
        </View>
        <View style={styles.headerRight}>
          <Text style={styles.headerTime}>{formatTime(time)}  UTC-4</Text>
          <View style={styles.avatarCircle}>
            <Text style={{ fontSize: 14 }}>👤</Text>
          </View>
          <Text style={styles.adminText}>Admin</Text>
        </View>
      </View>

      {/* Security Alert */}
      <View style={styles.alertBanner}>
        <Text style={styles.alertIcon}>⚠</Text>
        <View style={{ flex: 1 }}>
          <Text style={styles.alertTitle}>Security Alert Detected</Text>
          <Text style={styles.alertSub}>Unusual access pattern detected on Device ESP32-01. Last failed attempt: 2 minutes ago.</Text>
        </View>
        <TouchableOpacity>
          <Text style={styles.alertClose}>✕</Text>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false}>

        {/* Page title */}
        <View style={styles.titleRow}>
          <View>
            <Text style={styles.pageTitle}>ESP32 DEVICE STATUS</Text>
            <Text style={styles.pageSubtitle}>Real-time monitoring and control dashboard</Text>
          </View>
          <View style={styles.systemStatus}>
            <Text style={styles.systemStatusLabel}>SYSTEM STATUS</Text>
            <Text style={styles.systemStatusValue}>OPERATIONAL</Text>
          </View>
        </View>

        {/* Device Cards */}
        {devices.map((d) => (
          <DeviceCard key={d.id} device={d} />
        ))}

        {/* MQTT Broker Status */}
        <View style={styles.mqttCard}>
          <View style={styles.mqttCardHeader}>
            <Text style={styles.mqttCardTitle}>MQTT BROKER STATUS</Text>
            <View style={styles.mqttConnected}>
              <View style={[styles.statusDot, { backgroundColor: COLORS.green }]} />
              <Text style={styles.mqttConnectedText}>CONNECTED</Text>
            </View>
          </View>
          <View style={styles.mqttStats}>
            <View style={styles.mqttStat}>
              <Text style={[styles.mqttStatNum, { color: COLORS.white }]}>47</Text>
              <Text style={styles.mqttStatLabel}>MSG/SEC</Text>
            </View>
            <View style={styles.mqttStat}>
              <Text style={[styles.mqttStatNum, { color: COLORS.teal }]}>3</Text>
              <Text style={styles.mqttStatLabel}>ACTIVE DEVICES</Text>
            </View>
            <View style={styles.mqttStat}>
              <Text style={[styles.mqttStatNum, { color: COLORS.amber }]}>12</Text>
              <Text style={styles.mqttStatLabel}>TOPICS</Text>
            </View>
            <View style={styles.mqttStat}>
              <Text style={[styles.mqttStatNum, { color: COLORS.green }]}>99.6%</Text>
              <Text style={styles.mqttStatLabel}>UPTIME</Text>
            </View>
          </View>
        </View>

        {/* Quick Actions */}
        <View style={styles.actionsRow}>
          <TouchableOpacity
            style={[styles.actionBtn, { borderColor: COLORS.purple, backgroundColor: COLORS.purple + '18' }]}
            onPress={() => navigation?.navigate('VeinScan')}
          >
            <Text style={styles.actionIcon}>✋</Text>
            <Text style={[styles.actionLabel, { color: COLORS.purple }]}>START VEIN SCAN</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionBtn, { borderColor: COLORS.teal, backgroundColor: COLORS.teal + '14' }]}
            onPress={() => navigation?.navigate('AccessHistory')}
          >
            <Text style={styles.actionIcon}>🕐</Text>
            <Text style={[styles.actionLabel, { color: COLORS.teal }]}>VIEW HISTORY</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionBtn, { borderColor: COLORS.amber, backgroundColor: COLORS.amber + '18' }]}
            onPress={() => navigation?.navigate('UserManagement')}
          >
            <Text style={styles.actionIcon}>👥</Text>
            <Text style={[styles.actionLabel, { color: COLORS.amber }]}>MANAGE USERS</Text>
          </TouchableOpacity>
        </View>

        <View style={{ height: 24 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: COLORS.bg },

  // Header
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: COLORS.headerBg, paddingHorizontal: 14, paddingTop: 44, paddingBottom: 10,
    borderBottomWidth: 1, borderBottomColor: COLORS.cardBorder,
  },
  headerLeft: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  logoVein: { color: '#fff', fontWeight: '900', fontSize: 17, letterSpacing: 1 },
  logoGuard: { color: COLORS.teal, fontWeight: '900', fontSize: 17, letterSpacing: 1, marginRight: 8 },
  mqttBadge: {
    flexDirection: 'row', alignItems: 'center',
    borderWidth: 1, borderColor: COLORS.green, borderRadius: 20,
    paddingHorizontal: 8, paddingVertical: 3, gap: 4,
  },
  mqttText: { color: COLORS.green, fontSize: 9, fontWeight: '700', letterSpacing: 0.5 },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  headerTime: { color: COLORS.textDim, fontSize: 9 },
  avatarCircle: {
    width: 28, height: 28, borderRadius: 14,
    backgroundColor: COLORS.cardBorder, alignItems: 'center', justifyContent: 'center',
  },
  adminText: { color: COLORS.text, fontSize: 11 },

  // Alert
  alertBanner: {
    flexDirection: 'row', alignItems: 'flex-start',
    backgroundColor: '#2a0d00', borderBottomWidth: 1, borderBottomColor: '#7a2000',
    paddingHorizontal: 14, paddingVertical: 10, gap: 10,
  },
  alertIcon: { color: COLORS.amber, fontSize: 16, marginTop: 1 },
  alertTitle: { color: COLORS.amber, fontSize: 12, fontWeight: '700' },
  alertSub: { color: '#c07030', fontSize: 10, marginTop: 2, lineHeight: 14 },
  alertClose: { color: COLORS.textDim, fontSize: 14, marginTop: 1 },

  scroll: { flex: 1, paddingHorizontal: 14 },

  // Title
  titleRow: {
    flexDirection: 'row', alignItems: 'flex-start',
    justifyContent: 'space-between', marginTop: 14, marginBottom: 12,
  },
  pageTitle: { color: COLORS.white, fontSize: 18, fontWeight: '900', letterSpacing: 1 },
  pageSubtitle: { color: COLORS.textDim, fontSize: 10, marginTop: 2 },
  systemStatus: { alignItems: 'flex-end' },
  systemStatusLabel: { color: COLORS.textDim, fontSize: 8, letterSpacing: 1 },
  systemStatusValue: { color: COLORS.green, fontSize: 11, fontWeight: '800', marginTop: 2 },

  // Device Card
  deviceCard: {
    backgroundColor: COLORS.cardBg, borderRadius: 10, borderWidth: 1,
    padding: 14, marginBottom: 10,
  },
  deviceHeader: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10,
  },
  deviceId: { fontSize: 15, fontWeight: '900', letterSpacing: 1 },
  deviceRole: { color: COLORS.textDim, fontSize: 9, letterSpacing: 1, marginTop: 2 },
  deviceStats: {},
  deviceRow: {
    flexDirection: 'row', justifyContent: 'space-between',
    paddingVertical: 6, borderBottomWidth: 1, borderBottomColor: COLORS.cardBorder + '88',
  },
  deviceLabel: { color: COLORS.textDim, fontSize: 11 },
  deviceValue: { color: COLORS.white, fontSize: 11, fontWeight: '600' },

  // Dot
  statusDot: { width: 9, height: 9, borderRadius: 4.5 },

  // MQTT Card
  mqttCard: {
    backgroundColor: COLORS.cardBg, borderRadius: 10, borderWidth: 1,
    borderColor: COLORS.cardBorder, padding: 14, marginBottom: 12,
  },
  mqttCardHeader: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14,
  },
  mqttCardTitle: { color: COLORS.teal, fontSize: 11, fontWeight: '800', letterSpacing: 2 },
  mqttConnected: { flexDirection: 'row', alignItems: 'center', gap: 5 },
  mqttConnectedText: { color: COLORS.green, fontSize: 10, fontWeight: '700' },
  mqttStats: { flexDirection: 'row', justifyContent: 'space-between' },
  mqttStat: { alignItems: 'center', flex: 1 },
  mqttStatNum: { fontSize: 20, fontWeight: '900' },
  mqttStatLabel: { color: COLORS.textDim, fontSize: 8, letterSpacing: 0.5, marginTop: 3, textAlign: 'center' },

  // Actions
  actionsRow: { flexDirection: 'row', gap: 8, marginBottom: 12 },
  actionBtn: {
    flex: 1, borderRadius: 10, borderWidth: 1,
    paddingVertical: 18, alignItems: 'center', justifyContent: 'center',
  },
  actionIcon: { fontSize: 22, marginBottom: 6 },
  actionLabel: { fontSize: 9, fontWeight: '800', letterSpacing: 0.5, textAlign: 'center' },
});
