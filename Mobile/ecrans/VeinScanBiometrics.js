import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  StatusBar,
  Dimensions,
} from 'react-native';

const { width } = Dimensions.get('window');

const COLORS = {
  bg: '#080e1a',
  cardBg: '#0d1b2e',
  cardBorder: '#1a3a5c',
  green: '#00ff88',
  teal: '#00e5ff',
  amber: '#e6a020',
  red: '#ff3d5a',
  yellow: '#f0e020',
  text: '#b8cfe0',
  textDim: '#4a6a8a',
  white: '#ffffff',
  headerBg: '#0a1525',
  inputBg: '#091525',
  greenDark: '#002a1a',
  redDark: '#2a0010',
  amberDark: '#2a1500',
};

function QualityBar({ label, value, color }) {
  return (
    <View style={styles.qualityRow}>
      <Text style={styles.qualityLabel}>{label}</Text>
      <Text style={[styles.qualityVal, { color }]}>{value}%</Text>
      <View style={styles.qualityBg}>
        <View style={[styles.qualityFill, { width: `${value}%`, backgroundColor: color }]} />
      </View>
    </View>
  );
}

export default function VeinScanBiometrics() {
  const [userId, setUserId] = useState('');
  const [scanning, setScanning] = useState(false);

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
          <Text style={styles.headerTime}>12:32:00   UTC+0</Text>
          <View style={styles.avatarCircle}><Text>👤</Text></View>
          <Text style={styles.adminText}>Admin</Text>
        </View>
      </View>
      {/* Dropdown */}
      <View style={styles.dropdown}>
        <Text style={styles.dropdownText}>Vein Scan (Biometric HUD)</Text>
        <Text style={styles.dropdownArrow}>▼</Text>
      </View>
      {/* MQTT scroll ticker */}
      <View style={styles.ticker}>
        <Text style={styles.tickerText}>MQTT: Publishing scan request...   Topic: veinGuard-scan/request   Device: ESP32-21</Text>
      </View>

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false}>
        {/* Title */}
        <View style={styles.titleRow}>
          <View>
            <Text style={styles.pageTitle}>BIOMETRIC SCANNER HUD</Text>
            <Text style={styles.pageSubtitle}>Position hand over scanner for vein pattern capture</Text>
          </View>
          <View style={styles.scannerStatus}>
            <Text style={styles.scannerStatusText}>SCANNER STATUS</Text>
            <Text style={[styles.scannerStatusValue, { color: COLORS.green }]}>ACTIVE</Text>
          </View>
        </View>

        {/* Camera Preview */}
        <View style={styles.card}>
          <View style={styles.cameraHeader}>
            <Text style={styles.cameraTitle}>CAMERA PREVIEW</Text>
            <View style={styles.recBadge}>
              <View style={styles.recDot} />
              <Text style={styles.recText}>REC</Text>
            </View>
          </View>
          <View style={styles.cameraBox}>
            {/* Scan overlay */}
            <View style={styles.scanCornerTL} />
            <View style={styles.scanCornerTR} />
            <View style={styles.scanCornerBL} />
            <View style={styles.scanCornerBR} />
            {/* Circular scanner */}
            <View style={styles.scanCircle}>
              <Text style={styles.scanCircleIcon}>👋</Text>
            </View>
            {/* Target reticle */}
            <View style={styles.reticle}>
              <View style={styles.reticleH} />
              <View style={styles.reticleV} />
            </View>
          </View>
          <View style={styles.cameraFooter}>
            <Text style={styles.cameraFooterLeft}>POSITION HAND ABOVE SCANNER{'\n'}FRAME: 1920x1080</Text>
            <Text style={styles.cameraFooterRight}>IR ACTIVE{'\n'}FPS: 30</Text>
          </View>
        </View>

        {/* Scan Quality */}
        <View style={styles.card}>
          <Text style={styles.sectionTitle}>SCAN QUALITY</Text>
          <QualityBar label="Signal Strength" value={87} color={COLORS.green} />
          <QualityBar label="Pattern Clarity" value={64} color={COLORS.amber} />
          <QualityBar label="Alignment" value={92} color={COLORS.teal} />
        </View>

        {/* Scan Controls */}
        <View style={styles.card}>
          <Text style={styles.sectionTitle}>SCAN CONTROLS</Text>
          <TouchableOpacity
            style={[styles.ctrlBtn, styles.startBtn]}
            onPress={() => setScanning(true)}
          >
            <Text style={styles.ctrlBtnText}>▶ START SCAN</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.ctrlBtn, styles.stopBtn]}
            onPress={() => setScanning(false)}
          >
            <Text style={[styles.ctrlBtnText, { color: COLORS.red }]}>■ STOP SCAN</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.ctrlBtn, styles.irBtn]}>
            <Text style={[styles.ctrlBtnText, { color: COLORS.amber }]}>💡 TOGGLE IR</Text>
          </TouchableOpacity>
        </View>

        {/* MQTT Status */}
        <View style={styles.card}>
          <Text style={styles.sectionTitle}>MQTT STATUS</Text>
          <View style={styles.mqttRow}><Text style={styles.mqttLabel}>Connection</Text><Text style={[styles.mqttVal, { color: COLORS.green }]}>ONLINE</Text></View>
          <View style={styles.mqttRow}><Text style={styles.mqttLabel}>Last Publish</Text><Text style={styles.mqttVal}>2s ago</Text></View>
          <View style={styles.mqttRow}><Text style={styles.mqttLabel}>Topic</Text><Text style={styles.mqttVal}>scan/request</Text></View>
        </View>

        {/* Manual ID Entry */}
        <View style={styles.card}>
          <Text style={styles.sectionTitle}>MANUAL ID ENTRY (TESTING MODE)</Text>
          <Text style={styles.manualLabel}>User ID</Text>
          <View style={styles.manualRow}>
            <TextInput
              style={styles.manualInput}
              placeholder="Enter User ID for testing"
              placeholderTextColor={COLORS.textDim}
              value={userId}
              onChangeText={setUserId}
            />
            <TouchableOpacity style={styles.testBtn}>
              <Text style={styles.testBtnText}>TEST ACCESS</Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* Privacy notice */}
        <View style={styles.privacyCard}>
          <Text style={styles.privacyIcon}>🔒</Text>
          <Text style={styles.privacyText}>
            <Text style={styles.privacyBold}>Privacy Notice{'\n'}</Text>
            Biometric data is processed locally and encrypted before transmission. No vein patterns are stored permanently. All scan data is automatically purged after authentication.
          </Text>
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
  logoVein: { color: '#fff', fontWeight: '900', fontSize: 17 },
  logoGuard: { color: COLORS.teal, fontWeight: '900', fontSize: 17, marginRight: 10 },
  mqttBadge: { flexDirection: 'row', alignItems: 'center', borderWidth: 1, borderColor: COLORS.green, borderRadius: 20, paddingHorizontal: 8, paddingVertical: 3 },
  mqttDot: { width: 7, height: 7, borderRadius: 3.5, backgroundColor: COLORS.green, marginRight: 4 },
  mqttText: { color: COLORS.green, fontSize: 9, fontWeight: '700' },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  headerTime: { color: COLORS.textDim, fontSize: 10 },
  avatarCircle: { width: 28, height: 28, borderRadius: 14, backgroundColor: COLORS.cardBorder, alignItems: 'center', justifyContent: 'center' },
  adminText: { color: COLORS.text, fontSize: 11 },
  dropdown: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    marginHorizontal: 14, marginTop: 8, marginBottom: 0, padding: 10,
    backgroundColor: COLORS.cardBg, borderRadius: 8, borderWidth: 1, borderColor: COLORS.cardBorder,
  },
  dropdownText: { color: COLORS.teal, fontSize: 12 },
  dropdownArrow: { color: COLORS.textDim, fontSize: 10 },
  ticker: {
    backgroundColor: '#050d18', paddingHorizontal: 14, paddingVertical: 6,
    borderBottomWidth: 1, borderBottomColor: COLORS.cardBorder,
  },
  tickerText: { color: COLORS.green, fontSize: 10 },
  scroll: { flex: 1, paddingHorizontal: 14 },
  titleRow: { flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'space-between', marginTop: 12, marginBottom: 12 },
  pageTitle: { color: COLORS.white, fontSize: 18, fontWeight: '900', letterSpacing: 1, flex: 1 },
  pageSubtitle: { color: COLORS.textDim, fontSize: 10, marginTop: 3 },
  scannerStatus: { alignItems: 'flex-end', marginLeft: 8 },
  scannerStatusText: { color: COLORS.textDim, fontSize: 9, letterSpacing: 1 },
  scannerStatusValue: { fontSize: 12, fontWeight: '800', letterSpacing: 1, marginTop: 2 },
  card: { backgroundColor: COLORS.cardBg, borderRadius: 10, borderWidth: 1, borderColor: COLORS.cardBorder, padding: 14, marginBottom: 12 },
  sectionTitle: { color: COLORS.teal, fontSize: 11, fontWeight: '800', letterSpacing: 2, marginBottom: 12 },
  cameraHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
  cameraTitle: { color: COLORS.teal, fontSize: 11, fontWeight: '800', letterSpacing: 2 },
  recBadge: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  recDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: COLORS.red },
  recText: { color: COLORS.red, fontSize: 10, fontWeight: '700' },
  cameraBox: {
    height: 200, backgroundColor: '#050d18',
    borderRadius: 8, borderWidth: 1, borderColor: COLORS.teal,
    position: 'relative', alignItems: 'center', justifyContent: 'center',
    overflow: 'hidden', marginBottom: 8,
  },
  scanCornerTL: { position: 'absolute', top: 8, left: 8, width: 20, height: 20, borderTopWidth: 2, borderLeftWidth: 2, borderColor: COLORS.teal },
  scanCornerTR: { position: 'absolute', top: 8, right: 8, width: 20, height: 20, borderTopWidth: 2, borderRightWidth: 2, borderColor: COLORS.teal },
  scanCornerBL: { position: 'absolute', bottom: 8, left: 8, width: 20, height: 20, borderBottomWidth: 2, borderLeftWidth: 2, borderColor: COLORS.teal },
  scanCornerBR: { position: 'absolute', bottom: 8, right: 8, width: 20, height: 20, borderBottomWidth: 2, borderRightWidth: 2, borderColor: COLORS.teal },
  scanCircle: {
    width: 100, height: 100, borderRadius: 50,
    borderWidth: 2, borderColor: COLORS.teal,
    alignItems: 'center', justifyContent: 'center',
    backgroundColor: '#0a1e35',
  },
  scanCircleIcon: { fontSize: 40 },
  reticle: { position: 'absolute', width: 30, height: 30, top: 80, left: 100 },
  reticleH: { position: 'absolute', top: 15, left: 0, right: 0, height: 1, backgroundColor: COLORS.teal },
  reticleV: { position: 'absolute', left: 15, top: 0, bottom: 0, width: 1, backgroundColor: COLORS.teal },
  cameraFooter: { flexDirection: 'row', justifyContent: 'space-between' },
  cameraFooterLeft: { color: COLORS.textDim, fontSize: 9, lineHeight: 14 },
  cameraFooterRight: { color: COLORS.green, fontSize: 9, textAlign: 'right', lineHeight: 14 },
  qualityRow: { marginBottom: 10 },
  qualityLabel: { color: COLORS.text, fontSize: 11, marginBottom: 4 },
  qualityVal: { fontSize: 10, fontWeight: '700', textAlign: 'right', marginBottom: 2 },
  qualityBg: { height: 8, backgroundColor: '#050d18', borderRadius: 4, overflow: 'hidden' },
  qualityFill: { height: '100%', borderRadius: 4 },
  ctrlBtn: { borderRadius: 8, paddingVertical: 14, alignItems: 'center', marginBottom: 10, borderWidth: 1 },
  startBtn: { backgroundColor: COLORS.greenDark, borderColor: COLORS.green },
  stopBtn: { backgroundColor: COLORS.redDark, borderColor: COLORS.red },
  irBtn: { backgroundColor: COLORS.amberDark, borderColor: COLORS.amber },
  ctrlBtnText: { color: COLORS.green, fontSize: 13, fontWeight: '800', letterSpacing: 2 },
  mqttRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 6, borderBottomWidth: 1, borderBottomColor: COLORS.cardBorder },
  mqttLabel: { color: COLORS.textDim, fontSize: 12 },
  mqttVal: { color: COLORS.white, fontSize: 12, fontWeight: '600' },
  manualLabel: { color: COLORS.textDim, fontSize: 11, marginBottom: 8 },
  manualRow: { flexDirection: 'row', gap: 8 },
  manualInput: { flex: 1, backgroundColor: COLORS.inputBg, borderWidth: 1, borderColor: COLORS.cardBorder, borderRadius: 6, paddingHorizontal: 10, paddingVertical: 10, color: COLORS.white, fontSize: 12 },
  testBtn: { backgroundColor: COLORS.amberDark, borderWidth: 1, borderColor: COLORS.amber, borderRadius: 6, paddingHorizontal: 14, justifyContent: 'center' },
  testBtnText: { color: COLORS.amber, fontSize: 11, fontWeight: '800' },
  privacyCard: { flexDirection: 'row', backgroundColor: '#080f1e', borderRadius: 8, borderWidth: 1, borderColor: COLORS.cardBorder, padding: 12, marginBottom: 12 },
  privacyIcon: { fontSize: 16, marginRight: 10, marginTop: 2 },
  privacyText: { flex: 1, color: COLORS.textDim, fontSize: 10, lineHeight: 16 },
  privacyBold: { color: COLORS.text, fontWeight: '700' },
});
