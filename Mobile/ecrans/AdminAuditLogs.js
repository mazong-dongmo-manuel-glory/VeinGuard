import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  StatusBar,
  Platform,
  FlatList,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { useFocusEffect } from '@react-navigation/native';
import { COLORS, GRADIENTS } from '../theme';
import { useMqttStore } from '../store/mqttStore';
import { useAuthStore } from '../store/authStore';

function levelColor(level) {
  const colors = {
    CRITICAL: COLORS.neonRed,
    HIGH: COLORS.neonAmber,
    MEDIUM: COLORS.neonCyan,
    LOW: COLORS.neonGreen,
    INFO: COLORS.textDim,
  };
  return colors[level] || COLORS.textDim;
}

function translateLevel(level, t) {
  const labels = {
    CRITICAL: t('auditLogs.critical'),
    HIGH: t('auditLogs.high'),
    MEDIUM: t('auditLogs.medium'),
    LOW: t('auditLogs.low'),
    INFO: t('auditLogs.info'),
  };
  return labels[level] || level;
}

function SummaryCell({ value, label, color }) {
  return (
    <View style={styles.summaryCell}>
      <Text style={[styles.summaryVal, { color }]}>{value}</Text>
      <Text style={styles.summaryTitle}>{label}</Text>
    </View>
  );
}

function LogItem({ item, compact, showTechnicalDetails }) {
  const { t } = useTranslation();
  const color = levelColor(item.level);

  return (
    <View style={[styles.logItem, compact && styles.logItemCompact]}>
      <BlurView intensity={10} tint="dark" style={styles.logInner}>
        <View style={[styles.logIndicator, { backgroundColor: color }]} />
        <View style={styles.logIconWrap}>
          <Ionicons name="shield-outline" size={18} color={color} />
        </View>
        <View style={styles.logBody}>
          <View style={styles.logHeader}>
            <View style={[styles.levelTag, { borderColor: color, backgroundColor: `${color}15` }]}>
              <Text style={[styles.levelTagText, { color }]}>{translateLevel(item.level, t)}</Text>
            </View>
            <Text style={styles.logTime}>{new Date(item.timestamp).toLocaleString()}</Text>
          </View>
          <Text style={styles.logTitle}>{String(item.title || t('auditLogs.securityAlert')).toUpperCase()}</Text>
          <Text style={styles.logDesc}>{item.description}</Text>
          {showTechnicalDetails && item.meta ? <Text style={styles.logMeta}>{item.meta}</Text> : null}
        </View>
      </BlurView>
    </View>
  );
}

export default function AdminAuditLogs({ navigation }) {
  const { t } = useTranslation();
  const [filter, setFilter] = useState('ALL');
  const [auditList, setAuditList] = useState([]);
  const [refreshing, setRefreshing] = useState(false);

  const isConnected = useMqttStore((state) => state.isConnected);
  const fetchAuditLogs = useMqttStore((state) => state.fetchAuditLogs);
  const preferences = useAuthStore((state) => state.preferences);
  const autoRefreshData = Boolean(preferences?.autoRefreshData);
  const compactLists = Boolean(preferences?.compactLists);
  const showTechnicalDetails = Boolean(preferences?.showTechnicalDetails);

  const loadAudit = useCallback(async () => {
    if (!isConnected) {
      setAuditList([]);
      return;
    }

    try {
      const list = await fetchAuditLogs();
      setAuditList(Array.isArray(list) ? list : []);
    } catch (err) {
      console.error('Failed to fetch audit logs:', err);
    }
  }, [fetchAuditLogs, isConnected]);

  useFocusEffect(
    useCallback(() => {
      loadAudit();
    }, [loadAudit]),
  );

  useEffect(() => {
    if (!autoRefreshData || !isConnected) {
      return undefined;
    }

    const interval = setInterval(() => {
      loadAudit();
    }, 12000);

    return () => clearInterval(interval);
  }, [autoRefreshData, isConnected, loadAudit]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadAudit();
    setRefreshing(false);
  };

  const filteredLogs = useMemo(
    () => auditList.filter((item) => filter === 'ALL' || item.level === filter),
    [auditList, filter],
  );

  const filters = ['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'];

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" />
      <LinearGradient colors={GRADIENTS.primary} style={StyleSheet.absoluteFill} />

      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation?.goBack()} style={styles.backBtn}>
          <Ionicons name="chevron-back" size={24} color={COLORS.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{t('auditLogs.title')}</Text>
        <TouchableOpacity style={styles.headerAction}>
          <Ionicons name="shield-checkmark-outline" size={22} color={COLORS.neonGreen} />
        </TouchableOpacity>
      </View>

      <FlatList
        data={filteredLogs}
        keyExtractor={(item, index) => String(item.id || index)}
        refreshing={refreshing}
        onRefresh={handleRefresh}
        extraData={{ compactLists, showTechnicalDetails, filter }}
        renderItem={({ item }) => (
          <LogItem item={item} compact={compactLists} showTechnicalDetails={showTechnicalDetails} />
        )}
        ListHeaderComponent={(
          <View>
            <View style={styles.summarySection}>
              <BlurView intensity={15} style={styles.summaryCard}>
                <Text style={styles.summaryLabel}>{t('auditLogs.threatSummary')}</Text>
                <View style={styles.summaryGrid}>
                  <SummaryCell
                    value={auditList.filter((item) => item.level === 'CRITICAL').length}
                    label={t('auditLogs.critical')}
                    color={COLORS.neonRed}
                  />
                  <SummaryCell
                    value={auditList.filter((item) => item.level === 'HIGH').length}
                    label={t('auditLogs.high')}
                    color={COLORS.neonAmber}
                  />
                  <SummaryCell
                    value={auditList.length}
                    label={t('auditLogs.total')}
                    color={COLORS.neonCyan}
                  />
                </View>
              </BlurView>
            </View>

            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.filterScroll} contentContainerStyle={styles.filterContent}>
              {filters.map((value) => {
                const label = value === 'ALL' ? t('accessHistory.all') : translateLevel(value, t);
                return (
                  <TouchableOpacity
                    key={value}
                    style={[styles.filterBtn, filter === value && styles.filterBtnActive]}
                    onPress={() => setFilter(value)}
                  >
                    <Text style={[styles.filterText, filter === value && styles.filterTextActive]}>{label}</Text>
                  </TouchableOpacity>
                );
              })}
            </ScrollView>
          </View>
        )}
        ListEmptyComponent={(
          <Text style={styles.emptyText}>
            {isConnected ? t('auditLogs.noEvents') : t('userManagement.gatewayDisconnected')}
          </Text>
        )}
        ListFooterComponent={(
          <View style={styles.integritySection}>
            <BlurView intensity={5} style={styles.integrityCard}>
              <Ionicons name="cube-outline" size={18} color={COLORS.neonGreen} />
              <Text style={styles.integrityText}>{t('auditLogs.immutableAuditTrail')}</Text>
            </BlurView>
          </View>
        )}
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      />
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
  headerTitle: { color: COLORS.white, fontSize: 14, fontWeight: '800', letterSpacing: 1.4, flexShrink: 1, textAlign: 'center' },
  headerAction: { width: 40, height: 40, alignItems: 'flex-end', justifyContent: 'center' },
  scroll: { flex: 1, paddingHorizontal: 20 },
  scrollContent: { paddingTop: 10, paddingBottom: 40 },
  summarySection: { marginBottom: 25 },
  summaryCard: {
    borderRadius: 24,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
    overflow: 'hidden',
  },
  summaryLabel: { color: COLORS.textDim, fontSize: 10, fontWeight: '900', letterSpacing: 2, marginBottom: 20, textAlign: 'center' },
  summaryGrid: { flexDirection: 'row', justifyContent: 'space-around', gap: 12 },
  summaryCell: { alignItems: 'center', flex: 1 },
  summaryVal: { fontSize: 24, fontWeight: '900', marginBottom: 5 },
  summaryTitle: { color: COLORS.textDim, fontSize: 8, fontWeight: '800', letterSpacing: 1, textAlign: 'center' },
  filterScroll: { marginBottom: 25 },
  filterContent: { gap: 10, paddingRight: 20 },
  filterBtn: { paddingHorizontal: 18, paddingVertical: 10, borderRadius: 12, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.1)' },
  filterBtnActive: { borderColor: COLORS.neonCyan, backgroundColor: 'rgba(0, 243, 255, 0.05)' },
  filterText: { color: COLORS.textDim, fontSize: 11, fontWeight: '800' },
  filterTextActive: { color: COLORS.neonCyan },
  emptyText: { color: COLORS.textDim, textAlign: 'center', marginTop: 20 },
  logItem: { marginBottom: 15, borderRadius: 20, overflow: 'hidden' },
  logItemCompact: { marginBottom: 10 },
  logInner: { flexDirection: 'row', borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.05)' },
  logIndicator: { width: 4 },
  logIconWrap: { width: 42, alignItems: 'center', justifyContent: 'center' },
  logBody: { flex: 1, padding: 18, minWidth: 0 },
  logHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10, gap: 10 },
  levelTag: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: 6, borderWidth: 1 },
  levelTagText: { fontSize: 9, fontWeight: '900', letterSpacing: 1 },
  logTime: { color: COLORS.textDim, fontSize: 10, fontWeight: '600', flexShrink: 1, textAlign: 'right' },
  logTitle: { color: COLORS.white, fontSize: 14, fontWeight: '800', letterSpacing: 0.5, marginBottom: 5 },
  logDesc: { color: COLORS.textSecondary, fontSize: 12, lineHeight: 18, marginBottom: 8 },
  logMeta: { color: COLORS.textDim, fontSize: 10, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
  integritySection: { marginTop: 20, alignItems: 'center' },
  integrityCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 15,
    borderWidth: 1,
    borderColor: 'rgba(57, 255, 20, 0.1)',
    overflow: 'hidden',
  },
  integrityText: { color: COLORS.neonGreen, fontSize: 9, fontWeight: '800', letterSpacing: 1, textAlign: 'center' },
});
