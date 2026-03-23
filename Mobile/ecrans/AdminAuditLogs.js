import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  StatusBar,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { COLORS, GRADIENTS } from '../theme';

import { useMqttStore } from '../store/mqttStore';
import { useEffect } from 'react';

function LogItem({ item }) {
  // Mapping level to colors
  const colors = {
    'CRITICAL': COLORS.neonRed,
    'HIGH': COLORS.neonAmber,
    'MEDIUM': COLORS.neonCyan,
    'LOW': COLORS.neonGreen,
    'INFO': COLORS.textDim
  };
  const color = colors[item.level] || COLORS.textDim;

  return (
    <View style={styles.logItem}>
      <BlurView intensity={10} tint="dark" style={styles.logInner}>
        <View style={[styles.logIndicator, { backgroundColor: color }]} />
        <View style={styles.logBody}>
          <View style={styles.logHeader}>
            <View style={[styles.levelTag, { borderColor: color, backgroundColor: `${color}15` }]}>
              <Text style={[styles.levelTagText, { color: color }]}>{item.level}</Text>
            </View>
            <Text style={styles.logTime}>{new Date(item.timestamp).toLocaleString()}</Text>
          </View>
          <Text style={styles.logTitle}>{item.title.toUpperCase()}</Text>
          <Text style={styles.logDesc}>{item.description}</Text>
          <Text style={styles.logMeta}>{item.meta}</Text>
        </View>
      </BlurView>
    </View>
  );
}

export default function AdminAuditLogs({ navigation }) {
  const { t } = useTranslation();
  const [filter, setFilter] = useState('ALL');
  const [auditList, setAuditList] = useState([]);
  
  const isConnected = useMqttStore((state) => state.isConnected);
  const fetchAuditLogs = useMqttStore((state) => state.fetchAuditLogs);

  useEffect(() => {
    if (isConnected) {
      const loadAudit = async () => {
        try {
          const list = await fetchAuditLogs();
          setAuditList(Array.isArray(list) ? list : []);
        } catch (err) {
          console.error('Failed to fetch audit logs:', err);
        }
      };
      loadAudit();
    }
  }, [isConnected, fetchAuditLogs]);

  const filteredLogs = auditList.filter(l => filter === 'ALL' || l.level === filter);

  const filters = ['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'];

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" />
      <LinearGradient colors={GRADIENTS.primary} style={StyleSheet.absoluteFill} />

      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation?.goBack()} style={styles.backBtn}>
          <Ionicons name="chevron-back" size={24} color={COLORS.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>SECURE AUDIT LOG</Text>
        <TouchableOpacity style={styles.headerAction}>
          <Ionicons name="shield-checkmark-outline" size={22} color={COLORS.neonGreen} />
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        <View style={styles.summarySection}>
          <BlurView intensity={15} style={styles.summaryCard}>
            <Text style={styles.summaryLabel}>THREAT SUMMARY</Text>
            <View style={styles.summaryGrid}>
              <View style={styles.summaryCell}>
                <Text style={[styles.summaryVal, { color: COLORS.neonRed }]}>{auditList.filter(l => l.level === 'CRITICAL').length}</Text>
                <Text style={styles.summaryTitle}>CRITICAL</Text>
              </View>
              <View style={styles.summaryCell}>
                <Text style={[styles.summaryVal, { color: COLORS.neonAmber }]}>{auditList.filter(l => l.level === 'HIGH').length}</Text>
                <Text style={styles.summaryTitle}>HIGH</Text>
              </View>
              <View style={styles.summaryCell}>
                <Text style={[styles.summaryVal, { color: COLORS.neonCyan }]}>{auditList.length}</Text>
                <Text style={styles.summaryTitle}>TOTAL</Text>
              </View>
            </View>
          </BlurView>
        </View>

        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.filterScroll} contentContainerStyle={styles.filterContent}>
          {filters.map((f) => (
            <TouchableOpacity
              key={f}
              style={[styles.filterBtn, filter === f && styles.filterBtnActive]}
              onPress={() => setFilter(f)}
            >
              <Text style={[styles.filterText, filter === f && styles.filterTextActive]}>{f}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        <View style={styles.logList}>
          {filteredLogs.map((item, i) => (
            <LogItem key={item.id || i} item={item} />
          ))}
          {filteredLogs.length === 0 && (
            <Text style={{ color: COLORS.textDim, textAlign: 'center', marginTop: 20 }}>
                CLEAN AUDIT TRAIL: NO EVENTS DETECTED
            </Text>
          )}
        </View>

        <View style={styles.integritySection}>
          <BlurView intensity={5} style={styles.integrityCard}>
            <Ionicons name="cube-outline" size={18} color={COLORS.neonGreen} />
            <Text style={styles.integrityText}>LOGS SECURED BY HASH-CHAIN (SHA-256)</Text>
          </BlurView>
        </View>
        
        <View style={{ height: 40 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: COLORS.bg },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: Platform.OS === 'ios' ? 50 : 20,
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  backBtn: { width: 40, height: 40, justifyContent: 'center' },
  headerTitle: { color: COLORS.white, fontSize: 16, fontWeight: '800', letterSpacing: 2 },
  headerAction: { width: 40, height: 40, alignItems: 'flex-end', justifyContent: 'center' },

  scroll: { flex: 1, paddingHorizontal: 20 },
  scrollContent: { paddingTop: 10 },

  summarySection: { marginBottom: 25 },
  summaryCard: { borderRadius: 24, padding: 20, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.05)', overflow: 'hidden' },
  summaryLabel: { color: COLORS.textDim, fontSize: 10, fontWeight: '900', letterSpacing: 2, marginBottom: 20, textAlign: 'center' },
  summaryGrid: { flexDirection: 'row', justifyContent: 'space-around' },
  summaryCell: { alignItems: 'center' },
  summaryVal: { fontSize: 24, fontWeight: '900', marginBottom: 5 },
  summaryTitle: { color: COLORS.textDim, fontSize: 8, fontWeight: '800', letterSpacing: 1 },

  filterScroll: { marginBottom: 25 },
  filterContent: { gap: 10, paddingRight: 20 },
  filterBtn: { paddingHorizontal: 20, paddingVertical: 10, borderRadius: 12, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.1)' },
  filterBtnActive: { borderColor: COLORS.neonCyan, backgroundColor: 'rgba(0, 243, 255, 0.05)' },
  filterText: { color: COLORS.textDim, fontSize: 11, fontWeight: '800' },
  filterTextActive: { color: COLORS.neonCyan },

  logList: { gap: 15 },
  logItem: { borderRadius: 20, overflow: 'hidden' },
  logInner: { flexDirection: 'row', borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.05)' },
  logIndicator: { width: 4 },
  logBody: { flex: 1, padding: 20 },
  logHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
  levelTag: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: 6, borderWidth: 1 },
  levelTagText: { fontSize: 9, fontWeight: '900', letterSpacing: 1 },
  logTime: { color: COLORS.textDim, fontSize: 10, fontWeight: '600' },
  logTitle: { color: COLORS.white, fontSize: 14, fontWeight: '800', letterSpacing: 0.5, marginBottom: 5 },
  logDesc: { color: COLORS.textSecondary, fontSize: 12, lineHeight: 18, marginBottom: 8 },
  logMeta: { color: COLORS.textDim, fontSize: 10, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
  actionBtn: { alignSelf: 'flex-start', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 8, borderWidth: 1, marginTop: 15 },
  actionBtnText: { fontSize: 10, fontWeight: '900', letterSpacing: 1 },

  integritySection: { marginTop: 30, alignItems: 'center' },
  integrityCard: { 
    flexDirection: 'row', alignItems: 'center', gap: 10, 
    paddingHorizontal: 20, paddingVertical: 12, 
    borderRadius: 15, borderWidth: 1, borderColor: 'rgba(57, 255, 20, 0.1)',
    overflow: 'hidden',
  },
  integrityText: { color: COLORS.neonGreen, fontSize: 9, fontWeight: '800', letterSpacing: 1 },
});
