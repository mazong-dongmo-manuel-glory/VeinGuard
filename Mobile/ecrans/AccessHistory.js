import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  StatusBar,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { COLORS, GRADIENTS } from '../theme';

import { useMqttStore } from '../store/mqttStore';
import { useEffect } from 'react';

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

function EventItem({ item, onPress }) {
  const statusColor = item.status === 'GRANTED' ? COLORS.neonGreen : (item.status === 'DENIED' ? COLORS.neonRed : COLORS.neonAmber);
  const name = item.username || 'UNKNOWN OPERATIVE';
  
  return (
    <TouchableOpacity onPress={onPress} activeOpacity={0.7} style={styles.eventItem}>
      <BlurView intensity={15} tint="dark" style={styles.eventInner}>
        <View style={[styles.statusLine, { backgroundColor: statusColor }]} />
        <View style={styles.eventBody}>
          <View style={styles.eventHeader}>
            <Text style={styles.eventName}>{name.toUpperCase()}</Text>
            <Text style={styles.eventTime}>{new Date(item.timestamp).toLocaleTimeString()}</Text>
          </View>
          <Text style={styles.eventSub}>{item.method.toUpperCase()} // PORTAL-01</Text>
          <View style={styles.eventFooter}>
            <StatusTag status={item.status} color={statusColor} />
            <Text style={[styles.eventScore, { color: statusColor }]}>MATCH: --</Text>
          </View>
        </View>
      </BlurView>
    </TouchableOpacity>
  );
}

export default function AccessHistory({ navigation }) {
  const { t } = useTranslation();
  const [search, setSearch] = useState('');
  const [logList, setLogList] = useState([]);
  
  const isConnected = useMqttStore((state) => state.isConnected);
  const fetchLogs = useMqttStore((state) => state.fetchLogs);

  useEffect(() => {
    if (isConnected) {
      const loadLogs = async () => {
        try {
          const list = await fetchLogs();
          setLogList(Array.isArray(list) ? list : []);
        } catch (err) {
          console.error('Failed to fetch logs:', err);
        }
      };
      loadLogs();
    }
  }, [isConnected, fetchLogs]);

  const filteredLogs = logList.filter(l => 
    (l.username || 'unknown').toLowerCase().includes(search.toLowerCase()) ||
    l.status.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" />
      <LinearGradient colors={GRADIENTS.primary} style={StyleSheet.absoluteFill} />

      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation?.goBack()} style={styles.backBtn}>
          <Ionicons name="chevron-back" size={24} color={COLORS.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{t('accessHistory.title')}</Text>
        <TouchableOpacity style={styles.headerAction}>
          <Ionicons name="download-outline" size={22} color={COLORS.neonCyan} />
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
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
          <StatCard label={t('accessHistory.granted')} value={logList.filter(l => l.status === 'GRANTED').length} color={COLORS.neonGreen} icon="checkmark-circle" />
          <StatCard label={t('accessHistory.denied')} value={logList.filter(l => l.status === 'DENIED').length} color={COLORS.neonRed} icon="close-circle" />
          <StatCard label={t('accessHistory.errors')} value="0" color={COLORS.neonAmber} icon="warning" />
        </View>

        <View style={styles.logSection}>
          <Text style={styles.sectionTitle}>{t('accessHistory.eventLog').toUpperCase()}</Text>
          {filteredLogs.map((item, i) => (
            <EventItem
              key={item.id || i}
              item={item}
              onPress={() => navigation?.navigate('AccessEvent', { event: item })}
            />
          ))}
          {filteredLogs.length === 0 && (
            <Text style={{ color: COLORS.textDim, textAlign: 'center', marginTop: 20 }}>
                NO VASCULAR EVENTS RECORDED
            </Text>
          )}
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
  headerTitle: { color: COLORS.white, fontSize: 18, fontWeight: '800', letterSpacing: 1 },
  headerAction: { width: 40, height: 40, alignItems: 'flex-end', justifyContent: 'center' },

  scroll: { flex: 1, paddingHorizontal: 20 },
  scrollContent: { paddingTop: 10 },
  
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

  logSection: { gap: 12 },
  sectionTitle: { color: COLORS.textDim, fontSize: 10, fontWeight: '900', letterSpacing: 2, marginBottom: 15 },

  eventItem: { marginBottom: 12, borderRadius: 20, overflow: 'hidden' },
  eventInner: { flexDirection: 'row', borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.05)' },
  statusLine: { width: 4 },
  eventBody: { flex: 1, padding: 15 },
  eventHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 5 },
  eventName: { color: COLORS.white, fontSize: 14, fontWeight: '800', letterSpacing: 0.5 },
  eventTime: { color: COLORS.textDim, fontSize: 11, fontWeight: '600' },
  eventSub: { color: COLORS.textSecondary, fontSize: 12, marginBottom: 12 },
  eventFooter: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  statusTag: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: 6, borderWidth: 1 },
  statusTagText: { fontSize: 9, fontWeight: '900', letterSpacing: 0.5 },
  eventScore: { fontSize: 12, fontWeight: '800' },
});
