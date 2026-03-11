import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Switch,
  StatusBar,
} from 'react-native';

const COLORS = {
  bg: '#080e1a',
  cardBg: '#0d1b2e',
  cardBorder: '#1a3a5c',
  green: '#00ff88',
  teal: '#00e5ff',
  amber: '#e6a020',
  red: '#ff3d5a',
  text: '#b8cfe0',
  textDim: '#4a6a8a',
  white: '#ffffff',
  headerBg: '#0a1525',
  inputBg: '#091525',
  greenDark: '#002a1a',
};

function SectionHeader({ icon, title, color = COLORS.amber }) {
  return (
    <View style={[styles.sectionHeader, { borderBottomColor: color }]}>
      <Text style={[styles.sectionIcon, { color }]}>{icon}</Text>
      <Text style={[styles.sectionTitle, { color }]}>{title}</Text>
    </View>
  );
}

function InputField({ label, value, onChangeText, placeholder, secure }) {
  return (
    <View style={styles.inputGroup}>
      <Text style={styles.inputLabel}>{label}</Text>
      <TextInput
        style={styles.textInput}
        value={value}
        onChangeText={onChangeText}
        placeholder={placeholder}
        placeholderTextColor={COLORS.textDim}
        secureTextEntry={secure}
      />
    </View>
  );
}

function QoSRow({ topic, description, qos }) {
  return (
    <View style={styles.qosRow}>
      <View style={styles.qosLeft}>
        <Text style={styles.qosTopic}>{topic}</Text>
        <Text style={styles.qosDesc}>{description}</Text>
      </View>
      <View style={styles.qosBadge}>
        <Text style={styles.qosBadgeText}>{qos}</Text>
      </View>
    </View>
  );
}

function DeviceCard({ name, status, firmware, uptime, signal, lastSeen }) {
  const statusColor = status === 'Online' ? COLORS.green : status === 'Offline' ? COLORS.red : COLORS.amber;
  return (
    <View style={styles.deviceCard}>
      <View style={styles.deviceHeader}>
        <View style={styles.deviceInfo}>
          <View style={[styles.deviceDot, { backgroundColor: statusColor }]} />
          <View>
            <Text style={styles.deviceName}>{name}</Text>
            <Text style={[styles.deviceStatus, { color: statusColor }]}>{status}</Text>
          </View>
        </View>
        <View style={styles.deviceActions}>
          <TouchableOpacity style={styles.diagBtn}><Text style={styles.diagBtnText}>Diagnostics</Text></TouchableOpacity>
          <TouchableOpacity style={styles.rebootBtn}><Text style={styles.rebootBtnText}>Reboot</Text></TouchableOpacity>
        </View>
      </View>
      <View style={styles.deviceStats}>
        <Text style={styles.deviceStat}>Firmware: <Text style={styles.deviceStatVal}>{firmware}</Text></Text>
        <Text style={styles.deviceStat}>Uptime: <Text style={[styles.deviceStatVal, { color: COLORS.green }]}>{uptime}</Text></Text>
        <Text style={styles.deviceStat}>Signal: <Text style={[styles.deviceStatVal, { color: COLORS.green }]}>{signal}</Text></Text>
        <Text style={styles.deviceStat}>Last Seen: <Text style={styles.deviceStatVal}>{lastSeen}</Text></Text>
      </View>
    </View>
  );
}

export default function SystemSetting() {
  const [broker, setBroker] = useState('mqtt.veinguard.local');
  const [port, setPort] = useState('8883');
  const [clientId, setClientId] = useState('veinguard-app-001');
  const [username, setUsername] = useState('veinguard.admin');
  const [password, setPassword] = useState('••••••');
  const [tls, setTls] = useState(true);
  const [cleanSession, setCleanSession] = useState(true);
  const [biometric, setBiometric] = useState(true);
  const [autoLock, setAutoLock] = useState(true);
  const [pin, setPin] = useState(true);
  const [sessionTimeout, setSessionTimeout] = useState('15');

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.headerBg} />
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Text style={styles.logoVein}>VEIN</Text>
          <Text style={styles.logoGuard}>GUARD</Text>
          <View style={styles.mqttBadge}>
            <View style={styles.mqttDot} />
            <Text style={styles.mqttText}>MQTT ONLINE</Text>
          </View>
        </View>
        <View style={styles.headerRight}>
          <Text style={styles.headerAdmin}>Admin</Text>
          <View style={styles.avatarCircle}><Text>👤</Text></View>
        </View>
      </View>
      <View style={styles.dropdown}>
        <Text style={styles.dropdownText}>System Settings (MQTT/VPN Config)</Text>
        <Text style={styles.dropdownArrow}>▼</Text>
      </View>

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false}>
        {/* Page title */}
        <View style={styles.pageTitleRow}>
          <Text style={styles.settingsIcon}>⚙</Text>
          <Text style={styles.pageTitle}>SYSTEM SETTINGS</Text>
        </View>
        <Text style={styles.pageSubtitle}>MQTT broker configuration, device management, and security settings</Text>

        {/* MQTT BROKER CONFIG */}
        <View style={styles.card}>
          <SectionHeader icon="📡" title="MQTT BROKER CONFIG" color={COLORS.teal} />
          <TouchableOpacity style={styles.testConnBtn}><Text style={styles.testConnText}>TEST CONNECTION</Text></TouchableOpacity>
          <View style={styles.halfRow}>
            <InputField label="BROKER HOST" value={broker} onChangeText={setBroker} placeholder="mqtt host" />
            <InputField label="PORT" value={port} onChangeText={setPort} placeholder="8883" />
          </View>
          <View style={styles.halfRow}>
            <InputField label="CLIENT ID" value={clientId} onChangeText={setClientId} />
            <InputField label="PROTOCOL" value="MQTT v5.0" onChangeText={() => {}} />
          </View>
          <View style={styles.halfRow}>
            <InputField label="USERNAME" value={username} onChangeText={setUsername} />
            <InputField label="PASSWORD" value={password} onChangeText={setPassword} secure />
          </View>
          <View style={styles.checkRow}>
            <TouchableOpacity style={styles.checkbox} onPress={() => setTls(!tls)}>
              <View style={[styles.checkBox, tls && styles.checkBoxActive]}>{tls && <Text style={styles.checkMark}>✓</Text>}</View>
              <Text style={styles.checkLabel}>Enable TLS/SSL</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.checkbox} onPress={() => setCleanSession(!cleanSession)}>
              <View style={[styles.checkBox, cleanSession && styles.checkBoxActive]}>{cleanSession && <Text style={styles.checkMark}>✓</Text>}</View>
              <Text style={styles.checkLabel}>Clean Session</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.halfRow}>
            <InputField label="KEEP ALIVE (S)" value="60" onChangeText={() => {}} />
            <InputField label="RECONNECT DELAY (S)" value="5" onChangeText={() => {}} />
            <InputField label="MAX RECONNECT (S)" value="30" onChangeText={() => {}} />
          </View>
        </View>

        {/* TOPIC MAPPING & QoS */}
        <View style={styles.card}>
          <SectionHeader icon="🟣" title="TOPIC MAPPING & QoS" color="#c000ff" />
          <QoSRow topic="veinguard/devices/+/status" description="Receive device status updates" qos="QoS 1" />
          <QoSRow topic="veinguard/scan/request" description="Biometric scan requests" qos="QoS 1" />
          <QoSRow topic="veinguard/scan/result" description="Scan result messages" qos="QoS 2" />
          <QoSRow topic="veinguard/enroll/+" description="User enrollment events" qos="QoS 2" />
          <QoSRow topic="veinguard/telemetry/+" description="Device telemetry and diagnostics" qos="QoS 0" />
        </View>

        {/* ESP32 DEVICE MANAGEMENT */}
        <View style={styles.card}>
          <SectionHeader icon="🔴" title="ESP32 DEVICE MANAGEMENT" color={COLORS.amber} />
          <DeviceCard name="ESP32-MAIN-C1" status="Online" firmware="v3.1" uptime="48:21m" signal="-67dBm" lastSeen="5s ago" />
          <DeviceCard name="ESP32-LAB-C2" status="Online" firmware="v3.1" uptime="12h 11m" signal="-72dBm" lastSeen="3s ago" />
          <DeviceCard name="ESP32-SRV-C3" status="Warning" firmware="v2.8" uptime="120h 48m" signal="-83dBm" lastSeen="5s ago" />
          <TouchableOpacity style={styles.pairBtn}><Text style={styles.pairBtnText}>+ PAIR NEW DEVICE</Text></TouchableOpacity>
        </View>

        {/* APP SECURITY */}
        <View style={styles.card}>
          <SectionHeader icon="🔴" title="APP SECURITY" color={COLORS.red} />
          <View style={styles.secRow}>
            <View>
              <Text style={styles.secLabel}>Biometric Login</Text>
              <Text style={styles.secDesc}>Use fingerprint or face to unlock app</Text>
            </View>
            <Switch value={biometric} onValueChange={setBiometric} trackColor={{ true: COLORS.green }} thumbColor={COLORS.white} />
          </View>
          <View style={styles.secRow}>
            <View>
              <Text style={styles.secLabel}>Auto Lock</Text>
              <Text style={styles.secDesc}>Lock on</Text>
            </View>
            <Switch value={autoLock} onValueChange={setAutoLock} trackColor={{ true: COLORS.green }} thumbColor={COLORS.white} />
          </View>
          <View style={styles.inputGroup}>
            <Text style={styles.inputLabel}>SESSION TIMEOUT (MIN)</Text>
            <TextInput style={styles.textInput} value={sessionTimeout} onChangeText={setSessionTimeout} keyboardType="numeric" />
          </View>
          <View style={styles.secRow}>
            <View>
              <Text style={styles.secLabel}>PIN Required</Text>
              <Text style={styles.secDesc}>Require PIN for critical actions</Text>
            </View>
            <Switch value={pin} onValueChange={setPin} trackColor={{ true: COLORS.green }} thumbColor={COLORS.white} />
          </View>
        </View>

        {/* STATUS */}
        <View style={styles.card}>
          <SectionHeader icon="📊" title="STATUS" color={COLORS.teal} />
          <View style={styles.statusRow}><Text style={styles.statusLabel}>MQTT Broker</Text><Text style={[styles.statusVal, { color: COLORS.green }]}>CONNECTED</Text></View>
          <View style={styles.statusRow}><Text style={styles.statusLabel}>Active Devices</Text><Text style={styles.statusVal}>2/3</Text></View>
          <View style={styles.statusRow}><Text style={styles.statusLabel}>Pending Auth</Text><Text style={styles.statusVal}>0</Text></View>
          <View style={styles.statusRow}><Text style={styles.statusLabel}>Network Latency</Text><Text style={[styles.statusVal, { color: COLORS.green }]}>12ms</Text></View>
        </View>

        {/* QUICK ACTIONS */}
        <View style={styles.card}>
          <SectionHeader icon="⚡" title="QUICK ACTIONS" color={COLORS.amber} />
          <TouchableOpacity style={styles.qaBtn}><Text style={styles.qaBtnText}>View All Settings</Text></TouchableOpacity>
          <TouchableOpacity style={[styles.qaBtn, { marginTop: 8, borderColor: COLORS.teal }]}><Text style={[styles.qaBtnText, { color: COLORS.teal }]}>Export Config</Text></TouchableOpacity>
          <TouchableOpacity style={[styles.qaBtn, { marginTop: 8, borderColor: COLORS.red }]}><Text style={[styles.qaBtnText, { color: COLORS.red }]}>Reset to Defaults</Text></TouchableOpacity>
        </View>

        <View style={{ height: 32 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: COLORS.bg },
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: COLORS.headerBg, paddingHorizontal: 16, paddingTop: 44, paddingBottom: 10,
    borderBottomWidth: 1, borderBottomColor: COLORS.cardBorder,
  },
  headerLeft: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  logoVein: { color: '#fff', fontWeight: '900', fontSize: 18 },
  logoGuard: { color: COLORS.teal, fontWeight: '900', fontSize: 18, marginRight: 10 },
  mqttBadge: { flexDirection: 'row', alignItems: 'center', borderWidth: 1, borderColor: COLORS.green, borderRadius: 20, paddingHorizontal: 8, paddingVertical: 3 },
  mqttDot: { width: 7, height: 7, borderRadius: 3.5, backgroundColor: COLORS.green, marginRight: 4 },
  mqttText: { color: COLORS.green, fontSize: 9, fontWeight: '700' },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  headerAdmin: { color: COLORS.text, fontSize: 12 },
  avatarCircle: { width: 34, height: 34, borderRadius: 17, backgroundColor: COLORS.cardBorder, alignItems: 'center', justifyContent: 'center' },
  dropdown: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    marginHorizontal: 14, marginVertical: 8, padding: 10,
    backgroundColor: COLORS.cardBg, borderRadius: 8, borderWidth: 1, borderColor: COLORS.cardBorder,
  },
  dropdownText: { color: COLORS.teal, fontSize: 12 },
  dropdownArrow: { color: COLORS.textDim, fontSize: 10 },
  scroll: { flex: 1, paddingHorizontal: 14 },
  pageTitleRow: { flexDirection: 'row', alignItems: 'center', marginTop: 8, marginBottom: 4 },
  settingsIcon: { color: COLORS.amber, fontSize: 22, marginRight: 10 },
  pageTitle: { color: COLORS.white, fontSize: 22, fontWeight: '900', letterSpacing: 2 },
  pageSubtitle: { color: COLORS.textDim, fontSize: 11, marginBottom: 12 },
  card: { backgroundColor: COLORS.cardBg, borderRadius: 10, borderWidth: 1, borderColor: COLORS.cardBorder, padding: 14, marginBottom: 12 },
  sectionHeader: { flexDirection: 'row', alignItems: 'center', paddingBottom: 10, marginBottom: 12, borderBottomWidth: 1 },
  sectionIcon: { fontSize: 16, marginRight: 8 },
  sectionTitle: { fontSize: 13, fontWeight: '800', letterSpacing: 1.5 },
  testConnBtn: { position: 'absolute', right: 14, top: 14, borderWidth: 1, borderColor: COLORS.teal, borderRadius: 6, paddingHorizontal: 10, paddingVertical: 5 },
  testConnText: { color: COLORS.teal, fontSize: 10, fontWeight: '700' },
  halfRow: { flexDirection: 'row', gap: 10, marginBottom: 6 },
  inputGroup: { flex: 1, marginBottom: 10 },
  inputLabel: { color: COLORS.textDim, fontSize: 9, letterSpacing: 1, marginBottom: 5, textTransform: 'uppercase' },
  textInput: { backgroundColor: COLORS.inputBg, borderWidth: 1, borderColor: COLORS.cardBorder, borderRadius: 6, paddingHorizontal: 10, paddingVertical: 8, color: COLORS.white, fontSize: 12 },
  checkRow: { flexDirection: 'row', gap: 16, marginBottom: 10 },
  checkbox: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  checkBox: { width: 16, height: 16, borderRadius: 3, borderWidth: 1, borderColor: COLORS.cardBorder, alignItems: 'center', justifyContent: 'center' },
  checkBoxActive: { backgroundColor: COLORS.teal, borderColor: COLORS.teal },
  checkMark: { color: COLORS.headerBg, fontSize: 10, fontWeight: '900' },
  checkLabel: { color: COLORS.text, fontSize: 11 },
  qosRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: COLORS.cardBorder },
  qosLeft: { flex: 1 },
  qosTopic: { color: COLORS.teal, fontSize: 11, fontFamily: 'monospace' },
  qosDesc: { color: COLORS.textDim, fontSize: 10, marginTop: 2 },
  qosBadge: { backgroundColor: '#1a0a40', borderWidth: 1, borderColor: '#7000cc', borderRadius: 4, paddingHorizontal: 8, paddingVertical: 3 },
  qosBadgeText: { color: '#c060ff', fontSize: 10, fontWeight: '700' },
  deviceCard: { backgroundColor: COLORS.inputBg, borderRadius: 8, borderWidth: 1, borderColor: COLORS.cardBorder, padding: 12, marginBottom: 8 },
  deviceHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 },
  deviceInfo: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  deviceDot: { width: 10, height: 10, borderRadius: 5 },
  deviceName: { color: COLORS.white, fontSize: 13, fontWeight: '700' },
  deviceStatus: { fontSize: 11 },
  deviceActions: { flexDirection: 'row', gap: 6 },
  diagBtn: { borderWidth: 1, borderColor: COLORS.teal, borderRadius: 5, paddingHorizontal: 8, paddingVertical: 4 },
  diagBtnText: { color: COLORS.teal, fontSize: 9, fontWeight: '700' },
  rebootBtn: { borderWidth: 1, borderColor: COLORS.amber, borderRadius: 5, paddingHorizontal: 8, paddingVertical: 4 },
  rebootBtnText: { color: COLORS.amber, fontSize: 9, fontWeight: '700' },
  deviceStats: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  deviceStat: { color: COLORS.textDim, fontSize: 10 },
  deviceStatVal: { color: COLORS.white, fontWeight: '600' },
  pairBtn: { borderWidth: 1, borderColor: COLORS.teal, borderRadius: 8, paddingVertical: 12, alignItems: 'center', marginTop: 6, backgroundColor: '#001a2a' },
  pairBtnText: { color: COLORS.teal, fontSize: 12, fontWeight: '700', letterSpacing: 1 },
  secRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: COLORS.cardBorder },
  secLabel: { color: COLORS.white, fontSize: 13, fontWeight: '600' },
  secDesc: { color: COLORS.textDim, fontSize: 10, marginTop: 2 },
  statusRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 7, borderBottomWidth: 1, borderBottomColor: COLORS.cardBorder },
  statusLabel: { color: COLORS.textDim, fontSize: 12 },
  statusVal: { color: COLORS.white, fontSize: 12, fontWeight: '700' },
  qaBtn: { borderWidth: 1, borderColor: COLORS.green, borderRadius: 8, paddingVertical: 12, alignItems: 'center' },
  qaBtnText: { color: COLORS.green, fontSize: 12, fontWeight: '700' },
});
