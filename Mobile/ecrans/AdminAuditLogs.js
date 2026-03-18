import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  View, Text, StyleSheet, ScrollView,
  TouchableOpacity, StatusBar,
} from 'react-native';

const COLORS = {
  bg: '#080e1a', cardBg: '#0d1b2e', cardBorder: '#1a3a5c',
  green: '#00ff88', teal: '#00e5ff', amber: '#e6a020',
  red: '#ff3d5a', purple: '#c000ff', blue: '#4080ff',
  text: '#b8cfe0', textDim: '#4a6a8a', white: '#ffffff',
  headerBg: '#0a1525', greenDark: '#002a1a', redDark: '#2a0010',
};

const logs = [
  {
    level: 'CRITICAL', levelColor: COLORS.red, dot: COLORS.red,
    title: 'CRITICAL SECURITY ALERT',
    desc: 'Multiple failed biometric attempts detected - Device ESP32-MAIN-01',
    meta: 'User: Unknown  ·  IP: 192.168.1.45  ·  Hash: 8a29c…',
    time: '2024-01-16 14:22:15 UTC',
    action: 'Escalate Incident',
  },
  {
    level: 'HIGH', levelColor: COLORS.amber, dot: COLORS.amber,
    title: 'HIGH - ROLE MODIFICATION',
    desc: 'Admin role granted to user: sarah.connor@veinguard.com',
    meta: 'Modified by: admin@veinguard.com  ·  Hash: 7bf4fa…',
    time: '2024-01-16 14:18:32 UTC',
    action: 'Add Note',
  },
  {
    level: 'MEDIUM', levelColor: '#f0a020', dot: '#f0a020',
    title: 'MEDIUM - CONFIG CHANGE',
    desc: 'MQTT broker configuration updated - TLS settings modified',
    meta: 'Changed by: admin@veinguard.com  ·  Session: 9f5e2d…',
    time: '2024-01-16 14:15:47 UTC',
    action: null,
  },
  {
    level: 'LOW', levelColor: COLORS.green, dot: COLORS.green,
    title: 'LOW - USER ENROLLMENT',
    desc: 'New user enrolled: john.doe@veinguard.com',
    meta: 'Enrolled by: admin@veinguard.com  ·  Device: ESP32-MAIN-01',
    time: '2024-01-16 14:12:23 UTC',
    action: null,
  },
  {
    level: 'INFO', levelColor: COLORS.blue, dot: COLORS.blue,
    title: 'INFO - AUTHENTICATION SUCCESS',
    desc: 'Admin login successful: admin@veinguard.com',
    meta: 'IP: 192.168.1.103  ·  Browser: Chrome/120.0  ·  Hash: 6d0c8c…',
    time: '2024-01-16 14:08:15 UTC',
    action: null,
  },
  {
    level: 'CRITICAL', levelColor: COLORS.red, dot: COLORS.red,
    title: 'CRITICAL - DEVICE TAMPER',
    desc: 'Physical tampering detected on ESP32-LAB-02 - Case opened',
    meta: 'Device: ESP32-LAB-02  ·  Location: Laboratory  ·  Hash: 3e5a2f…',
    time: '2024-01-16 13:45:22 UTC',
    action: 'Escalate Incident',
  },
];

function LogItem({ item }) {
  return (
    <View style={[styles.logItem, { borderLeftColor: item.dot }]}>
      <View style={styles.logHeader}>
        <View style={[styles.levelBadge, { backgroundColor: item.levelColor + '22', borderColor: item.levelColor }]}>
          <Text style={[styles.levelText, { color: item.levelColor }]}>{item.level}</Text>
        </View>
        <Text style={styles.logTime}>{item.time}</Text>
      </View>
      <Text style={styles.logTitle}>{item.title}</Text>
      <Text style={styles.logDesc}>{item.desc}</Text>
      <Text style={styles.logMeta}>{item.meta}</Text>
      {item.action && (
        <TouchableOpacity style={[styles.actionBtn, { borderColor: item.levelColor }]}>
          <Text style={[styles.actionBtnText, { color: item.levelColor }]}>⚠ {item.action}</Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

export default function AdminAuditLogs() {
  const { t } = useTranslation();
  const [filter, setFilter] = useState('All Events');

  const filters = ['All Events', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'];

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
            <Text style={styles.mqttText}>{t('login.mqttBadge')}</Text>
          </View>
        </View>
        <View style={styles.avatarCircle}><Text>👤</Text></View>
      </View>

      {/* Title */}
      <View style={styles.titleSection}>
        <View style={styles.titleRow}>
          <View style={styles.shieldBadge}><Text style={styles.shieldIcon}>🛡</Text></View>
          <View>
            <Text style={styles.pageTitle}>{t('auditLogs.title')}</Text>
            <Text style={styles.pageSubtitle}>{t('auditLogs.subtitle')}</Text>
          </View>
        </View>
        {/* Actions */}
        <View style={styles.actionsRow}>
          <TouchableOpacity style={styles.exportBtn}>
            <Text style={styles.exportBtnText}>{t('auditLogs.export')}</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.refreshBtn}>
            <Text style={styles.refreshBtnText}>{t('auditLogs.refresh')}</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Threat Summary */}
      <View style={styles.threatCard}>
        <Text style={styles.threatTitle}>{t('auditLogs.threatSummary')}</Text>
        <View style={styles.threatRow}>
          <Text style={styles.threatLabel}>{t('auditLogs.criticalAlerts')}</Text>
          <Text style={[styles.threatVal, { color: COLORS.red }]}>3</Text>
        </View>
        <View style={styles.threatRow}>
          <Text style={styles.threatLabel}>{t('auditLogs.highPriority')}</Text>
          <Text style={[styles.threatVal, { color: COLORS.amber }]}>12</Text>
        </View>
        <View style={styles.threatRow}>
          <Text style={styles.threatLabel}>{t('auditLogs.mediumPriority')}</Text>
          <Text style={[styles.threatVal, { color: '#f0a020' }]}>28</Text>
        </View>
        <View style={[styles.threatRow, { borderBottomWidth: 0 }]}>
          <Text style={styles.threatLabel}>{t('auditLogs.failedAttempts')}</Text>
          <Text style={[styles.threatVal, { color: COLORS.white }]}>47</Text>
        </View>
      </View>

      {/* Filter chips */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.filterScroll} contentContainerStyle={styles.filterContent}>
        {filters.map((f) => (
          <TouchableOpacity
            key={f}
            style={[styles.filterChip, filter === f && styles.filterChipActive]}
            onPress={() => setFilter(f)}
          >
            <Text style={[styles.filterChipText, filter === f && styles.filterChipTextActive]}>{f}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* Log List */}
      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false}>
        <View style={styles.auditCard}>
          <View style={styles.auditHeader}>
            <Text style={styles.auditTitle}>{t('auditLogs.immutableAuditTrail')}</Text>
            <View style={styles.cryptoBadge}>
              <Text style={styles.cryptoText}>{t('auditLogs.secured')}</Text>
            </View>
          </View>
          {logs.map((item, i) => (
            <LogItem key={i} item={item} />
          ))}
          <Text style={styles.paginationText}>Showing 6 of 1,247 events</Text>
          <View style={styles.paginationRow}>
            <TouchableOpacity style={styles.pageBtn}><Text style={styles.pageBtnText}>← Previous</Text></TouchableOpacity>
            <TouchableOpacity style={[styles.pageBtn, styles.pageBtnActive]}><Text style={[styles.pageBtnText, { color: COLORS.teal }]}>Next →</Text></TouchableOpacity>
          </View>
        </View>

        {/* Log Integrity */}
        <View style={styles.integrityCard}>
          <Text style={styles.integrityTitle}>🔵 LOG INTEGRITY</Text>
          {[['Blockchain Verified', true], ['Hash Chain Valid', true], ['Signature Valid', true]].map(([label, ok]) => (
            <View key={label} style={styles.integrityRow}>
              <Text style={styles.integrityLabel}>{label}</Text>
              <Text style={[styles.integrityVal, { color: ok ? COLORS.green : COLORS.red }]}>{ok ? '✓' : '✗'}</Text>
            </View>
          ))}
          <View style={styles.tamperProof}>
            <Text style={styles.tamperProofText}>{t('auditLogs.tamperProof')}</Text>
          </View>
        </View>

        <View style={{ height: 24 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: COLORS.bg },
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: COLORS.headerBg, paddingHorizontal: 14, paddingTop: 44, paddingBottom: 10,
    borderBottomWidth: 1, borderBottomColor: COLORS.cardBorder,
  },
  headerLeft: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  logoVein: { color: '#fff', fontWeight: '900', fontSize: 17, letterSpacing: 1 },
  logoGuard: { color: COLORS.teal, fontWeight: '900', fontSize: 17, letterSpacing: 1, marginRight: 8 },
  mqttBadge: { flexDirection: 'row', alignItems: 'center', borderWidth: 1, borderColor: COLORS.green, borderRadius: 20, paddingHorizontal: 8, paddingVertical: 3 },
  mqttDot: { width: 7, height: 7, borderRadius: 3.5, backgroundColor: COLORS.green, marginRight: 4 },
  mqttText: { color: COLORS.green, fontSize: 9, fontWeight: '700' },
  avatarCircle: { width: 30, height: 30, borderRadius: 15, backgroundColor: COLORS.cardBorder, alignItems: 'center', justifyContent: 'center' },

  titleSection: { padding: 14, borderBottomWidth: 1, borderBottomColor: COLORS.cardBorder },
  titleRow: { flexDirection: 'row', alignItems: 'flex-start', marginBottom: 10, gap: 10 },
  shieldBadge: { width: 38, height: 38, borderRadius: 10, backgroundColor: COLORS.red + '22', borderWidth: 1, borderColor: COLORS.red, alignItems: 'center', justifyContent: 'center' },
  shieldIcon: { fontSize: 18 },
  pageTitle: { color: COLORS.white, fontSize: 18, fontWeight: '900', letterSpacing: 1, lineHeight: 24 },
  pageSubtitle: { color: COLORS.textDim, fontSize: 10, marginTop: 2 },
  actionsRow: { flexDirection: 'row', gap: 8 },
  exportBtn: { borderWidth: 1, borderColor: COLORS.red, borderRadius: 6, paddingHorizontal: 14, paddingVertical: 8, backgroundColor: COLORS.red + '15' },
  exportBtnText: { color: COLORS.red, fontSize: 11, fontWeight: '700' },
  refreshBtn: { borderWidth: 1, borderColor: COLORS.teal, borderRadius: 6, paddingHorizontal: 14, paddingVertical: 8 },
  refreshBtnText: { color: COLORS.teal, fontSize: 11, fontWeight: '700' },

  threatCard: { marginHorizontal: 14, marginVertical: 10, backgroundColor: COLORS.cardBg, borderRadius: 10, borderWidth: 1, borderColor: COLORS.amber + '55', padding: 12 },
  threatTitle: { color: COLORS.amber, fontSize: 11, fontWeight: '800', letterSpacing: 1.5, marginBottom: 8 },
  threatRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 5, borderBottomWidth: 1, borderBottomColor: COLORS.cardBorder },
  threatLabel: { color: COLORS.text, fontSize: 12 },
  threatVal: { fontSize: 13, fontWeight: '900' },

  filterScroll: { maxHeight: 44 },
  filterContent: { paddingHorizontal: 12, paddingVertical: 6, gap: 6, alignItems: 'center' },
  filterChip: { borderRadius: 20, borderWidth: 1, borderColor: COLORS.cardBorder, paddingHorizontal: 12, paddingVertical: 5, backgroundColor: COLORS.cardBg },
  filterChipActive: { borderColor: COLORS.teal, backgroundColor: COLORS.teal + '20' },
  filterChipText: { color: COLORS.textDim, fontSize: 11, fontWeight: '600' },
  filterChipTextActive: { color: COLORS.teal },

  scroll: { flex: 1, paddingHorizontal: 14 },
  auditCard: { backgroundColor: COLORS.cardBg, borderRadius: 10, borderWidth: 1, borderColor: COLORS.cardBorder, padding: 12, marginTop: 10, marginBottom: 10 },
  auditHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  auditTitle: { color: COLORS.white, fontSize: 12, fontWeight: '800', letterSpacing: 1 },
  cryptoBadge: { borderWidth: 1, borderColor: COLORS.green, borderRadius: 4, paddingHorizontal: 8, paddingVertical: 3 },
  cryptoText: { color: COLORS.green, fontSize: 9, fontWeight: '700' },

  logItem: { borderLeftWidth: 3, paddingLeft: 10, marginBottom: 14, paddingBottom: 14, borderBottomWidth: 1, borderBottomColor: COLORS.cardBorder + '66' },
  logHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 },
  levelBadge: { borderRadius: 4, borderWidth: 1, paddingHorizontal: 8, paddingVertical: 2 },
  levelText: { fontSize: 9, fontWeight: '800', letterSpacing: 1 },
  logTime: { color: COLORS.textDim, fontSize: 9 },
  logTitle: { color: COLORS.white, fontSize: 12, fontWeight: '700', marginBottom: 3 },
  logDesc: { color: COLORS.text, fontSize: 11, marginBottom: 3 },
  logMeta: { color: COLORS.textDim, fontSize: 9, marginBottom: 5 },
  actionBtn: { alignSelf: 'flex-start', borderWidth: 1, borderRadius: 4, paddingHorizontal: 10, paddingVertical: 4 },
  actionBtnText: { fontSize: 10, fontWeight: '700' },

  paginationText: { color: COLORS.textDim, fontSize: 10, textAlign: 'center', marginTop: 8, marginBottom: 6 },
  paginationRow: { flexDirection: 'row', gap: 8 },
  pageBtn: { flex: 1, borderWidth: 1, borderColor: COLORS.cardBorder, borderRadius: 6, paddingVertical: 8, alignItems: 'center' },
  pageBtnActive: { borderColor: COLORS.teal },
  pageBtnText: { color: COLORS.textDim, fontSize: 11, fontWeight: '700' },

  integrityCard: { backgroundColor: COLORS.cardBg, borderRadius: 10, borderWidth: 1, borderColor: COLORS.cardBorder, padding: 14, marginBottom: 10 },
  integrityTitle: { color: COLORS.teal, fontSize: 11, fontWeight: '800', letterSpacing: 1.5, marginBottom: 10 },
  integrityRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 6, borderBottomWidth: 1, borderBottomColor: COLORS.cardBorder },
  integrityLabel: { color: COLORS.text, fontSize: 12 },
  integrityVal: { fontSize: 14, fontWeight: '800' },
  tamperProof: { marginTop: 10, alignItems: 'center' },
  tamperProofText: { color: COLORS.green, fontSize: 11, fontWeight: '800', letterSpacing: 2 },
});
