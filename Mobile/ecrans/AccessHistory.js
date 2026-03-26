import React, { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  StatusBar,
  Platform,
  FlatList,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { useFocusEffect } from '@react-navigation/native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { COLORS, GRADIENTS } from '../theme';
import { getAppErrorMessage } from '../services/appErrors';
import { useMqttStore } from '../store/mqttStore';
import { useAuthStore } from '../store/authStore';

function StatCard({ label, value, color, icon }) {
  return (
    <BlurView intensity={12} tint="dark" style={[styles.statCard, { borderColor: `${color}30` }]}>
      <Ionicons name={icon} size={18} color={color} style={styles.statIcon} />
      <Text style={[styles.statValue, { color }]}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </BlurView>
  );
}

function StatusTag({ status, color }) {
  return (
    <View style={[styles.statusTag, { borderColor: `${color}40`, backgroundColor: `${color}15` }]}>
      <Text style={[styles.statusTagText, { color }]}>{status}</Text>
    </View>
  );
}

function EventItem({ item, onPress, compact, showTechnicalDetails }) {
  const { t } = useTranslation();
  const statusColor = item.status === 'GRANTED' ? COLORS.neonGreen : (item.status === 'DENIED' ? COLORS.neonRed : COLORS.neonAmber);
  const statusLabel =
    item.status === 'GRANTED'
      ? t('accessHistory.granted')
      : item.status === 'DENIED'
        ? t('accessHistory.denied')
        : item.status;

  return (
    <TouchableOpacity onPress={onPress} activeOpacity={0.7} style={[styles.eventItem, compact && styles.eventItemCompact]}>
      <BlurView intensity={15} tint="dark" style={styles.eventInner}>
        <View style={[styles.statusLine, { backgroundColor: statusColor }]} />
        <View style={styles.eventIconWrap}>
          <Ionicons
            name={item.status === 'GRANTED' ? 'checkmark-circle-outline' : 'alert-circle-outline'}
            size={20}
            color={statusColor}
          />
        </View>
        <View style={styles.eventBody}>
          <View style={styles.eventHeader}>
            <Text numberOfLines={1} style={styles.eventName}>{(item.username || t('common.unknownUser')).toUpperCase()}</Text>
            <Text style={styles.eventTime}>{new Date(item.timestamp).toLocaleTimeString()}</Text>
          </View>
          {showTechnicalDetails ? (
            <Text numberOfLines={1} style={styles.eventSub}>{String(item.method || '').toUpperCase()} // PORTAIL-01</Text>
          ) : null}
          <View style={styles.eventFooter}>
            <StatusTag status={statusLabel} color={statusColor} />
            {showTechnicalDetails ? (
              <Text style={[styles.eventScore, { color: statusColor }]}>{t('accessDecision.confidence')}: {item.score ?? '--'}</Text>
            ) : null}
          </View>
        </View>
      </BlurView>
    </TouchableOpacity>
  );
}

export default function AccessHistory({ navigation }) {
  const { t } = useTranslation();
  const insets = useSafeAreaInsets();
  const [search, setSearch] = useState('');
  const [logList, setLogList] = useState([]);
  const [refreshing, setRefreshing] = useState(false);
  const [loadError, setLoadError] = useState(null);

  const isConnected = useMqttStore((state) => state.isConnected);
  const gatewayOnline = useMqttStore((state) => state.gatewayOnline);
  const fetchLogs = useMqttStore((state) => state.fetchLogs);
  const preferences = useAuthStore((state) => state.preferences);
  const autoRefreshData = Boolean(preferences?.autoRefreshData);
  const compactLists = Boolean(preferences?.compactLists);
  const showTechnicalDetails = Boolean(preferences?.showTechnicalDetails);

  const loadLogs = useCallback(async () => {
    if (!isConnected) {
      setLogList([]);
      setLoadError(t('userManagement.gatewayDisconnected'));
      return;
    }

    try {
      const list = await fetchLogs();
      setLogList(Array.isArray(list) ? list : []);
      setLoadError(null);
    } catch (err) {
      setLogList([]);
      setLoadError(getAppErrorMessage(t, err, 'userManagement.gatewayDisconnected'));
    }
  }, [isConnected, fetchLogs, t]);

  useFocusEffect(
    useCallback(() => {
      void loadLogs();
    }, [loadLogs]),
  );

  useEffect(() => {
    if (!autoRefreshData || !isConnected) {
      return undefined;
    }

    const interval = setInterval(() => {
      void loadLogs();
    }, 10000);

    return () => clearInterval(interval);
  }, [autoRefreshData, isConnected, loadLogs]);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await loadLogs();
    } finally {
      setRefreshing(false);
    }
  };

  const filteredLogs = logList.filter((log) =>
    `${log.username || ''} ${log.status || ''}`.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" />
      <LinearGradient colors={GRADIENTS.primary} style={StyleSheet.absoluteFill} />

      <View style={[styles.header, { paddingTop: Math.max(insets.top + 8, 20) }]}>
        <TouchableOpacity onPress={() => navigation?.goBack()} style={styles.backBtn}>
          <Ionicons name="chevron-back" size={24} color={COLORS.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{t('accessHistory.title')}</Text>
        <TouchableOpacity style={styles.headerAction} onPress={() => navigation?.navigate('AdminAuditLogs')}>
          <Ionicons name="shield-checkmark-outline" size={22} color={COLORS.neonCyan} />
        </TouchableOpacity>
      </View>

      <FlatList
        data={filteredLogs}
        keyExtractor={(item, index) => String(item.id || index)}
        refreshing={refreshing}
        onRefresh={handleRefresh}
        renderItem={({ item }) => (
          <EventItem
            item={item}
            onPress={() => navigation?.navigate('AccessEvent', { event: item })}
            compact={compactLists}
            showTechnicalDetails={showTechnicalDetails}
          />
        )}
        extraData={{ compactLists, showTechnicalDetails }}
        ListHeaderComponent={(
          <View>
            <View style={styles.searchSection}>
              <BlurView intensity={10} style={styles.searchBar}>
                <Ionicons name="search" size={18} color={COLORS.textDim} />
                <TextInput
                  style={styles.searchInput}
                  placeholder={t('accessHistory.searchPlaceholder')}
                  placeholderTextColor={COLORS.textDim}
                  value={search}
                  onChangeText={setSearch}
                />
              </BlurView>
            </View>

            <View style={styles.statsRow}>
              <StatCard label={t('accessHistory.granted')} value={logList.filter((l) => l.status === 'GRANTED').length} color={COLORS.neonGreen} icon="checkmark-circle" />
              <StatCard label={t('accessHistory.denied')} value={logList.filter((l) => l.status === 'DENIED').length} color={COLORS.neonRed} icon="close-circle" />
              <StatCard label={t('accessHistory.errors')} value="0" color={COLORS.neonAmber} icon="warning" />
            </View>

            <View style={styles.logSection}>
              <Text style={styles.sectionTitle}>{t('accessHistory.eventLog').toUpperCase()}</Text>
            </View>
          </View>
        )}
        ListEmptyComponent={(
          <Text style={styles.emptyText}>
            {loadError || ((gatewayOnline || isConnected) ? t('accessHistory.noEvents') : t('userManagement.gatewayDisconnected'))}
          </Text>
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
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  backBtn: { width: 40, height: 40, justifyContent: 'center' },
  headerTitle: { color: COLORS.white, fontSize: 18, fontWeight: '800', letterSpacing: 1 },
  headerAction: { width: 40, height: 40, alignItems: 'flex-end', justifyContent: 'center' },
  scroll: { flex: 1, paddingHorizontal: 20 },
  scrollContent: { paddingTop: 10, paddingBottom: 40 },
  searchSection: { marginBottom: 25 },
  searchBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 15,
    paddingVertical: 12,
    borderRadius: 15,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
    gap: 10,
    overflow: 'hidden',
  },
  searchInput: { flex: 1, color: COLORS.white, fontSize: 14, fontWeight: '500' },
  statsRow: { flexDirection: 'row', gap: 12, marginBottom: 30 },
  statCard: {
    flex: 1,
    padding: 15,
    borderRadius: 20,
    borderWidth: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.02)',
    overflow: 'hidden',
  },
  statIcon: { marginBottom: 10 },
  statValue: { fontSize: 24, fontWeight: '900', marginBottom: 4 },
  statLabel: { color: COLORS.textDim, fontSize: 8, fontWeight: '800', letterSpacing: 1 },
  logSection: { marginBottom: 10 },
  sectionTitle: { color: COLORS.textDim, fontSize: 10, fontWeight: '900', letterSpacing: 2, marginBottom: 15 },
  emptyText: { color: COLORS.textDim, textAlign: 'center', marginTop: 20 },
  eventItem: { marginBottom: 12, borderRadius: 20, overflow: 'hidden' },
  eventItemCompact: { marginBottom: 10 },
  eventInner: { flexDirection: 'row', borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.05)', alignItems: 'stretch' },
  statusLine: { width: 4 },
  eventIconWrap: { width: 44, justifyContent: 'center', alignItems: 'center' },
  eventBody: { flex: 1, padding: 15 },
  eventHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 5, gap: 10 },
  eventName: { color: COLORS.white, fontSize: 14, fontWeight: '800', letterSpacing: 0.5, flex: 1 },
  eventTime: { color: COLORS.textDim, fontSize: 11, fontWeight: '600' },
  eventSub: { color: COLORS.textSecondary, fontSize: 12, marginBottom: 12 },
  eventFooter: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 12 },
  statusTag: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: 6, borderWidth: 1 },
  statusTagText: { fontSize: 9, fontWeight: '900', letterSpacing: 0.5 },
  eventScore: { fontSize: 12, fontWeight: '800' },
});
