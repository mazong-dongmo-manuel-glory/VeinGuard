import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  StatusBar,
} from 'react-native';

const COLORS = {
  bg: '#080e1a',
  cardBg: '#0d1b2e',
  cardBorder: '#1a3a5c',
  green: '#00ff88',
  teal: '#00e5ff',
  red: '#ff3d5a',
  amber: '#e6a020',
  text: '#b8cfe0',
  textDim: '#4a6a8a',
  white: '#ffffff',
  headerBg: '#0a1525',
  rowBg: '#0b1928',
  greenDark: '#002a1a',
  redDark: '#2a0010',
};

const events = [
  { id: 'EVT-8743', name: 'John Mitchell', sub: 'ESP32-01 · DOOR-A12', score: '98.7%', time: '14:23:45', status: 'GRANTED', dot: COLORS.green },
  { id: 'EVT-8742', name: 'Unknown User', sub: 'ESP32-02 · DOOR-B05', score: '42.1%', time: '14:18:22', status: 'DENIED', dot: COLORS.red },
  { id: 'EVT-8741', name: 'Sarah Chen', sub: 'ESP32-01 · DOOR-A12', score: '95.3%', time: '14:15:08', status: 'GRANTED', dot: COLORS.green },
  { id: 'EVT-8740', name: 'Mike Rodriguez', sub: 'ESP32-03 · DOOR-C08', score: '--', time: '14:12:33', status: 'ERROR', dot: COLORS.amber },
  { id: 'EVT-8739', name: 'Alex Thompson', sub: 'ESP32-02 · DOOR-B05', score: '91.8%', time: '14:08:17', status: 'GRANTED', dot: COLORS.green },
];

function StatusBadge({ status }) {
  const colors = {
    GRANTED: { bg: COLORS.greenDark, text: COLORS.green, border: COLORS.green },
    DENIED: { bg: COLORS.redDark, text: COLORS.red, border: COLORS.red },
    ERROR: { bg: '#2a1800', text: COLORS.amber, border: COLORS.amber },
  };
  const c = colors[status] || colors.GRANTED;
  return (
    <View style={[styles.badge, { backgroundColor: c.bg, borderColor: c.border }]}>
      <Text style={[styles.badgeText, { color: c.text }]}>{status}</Text>
    </View>
  );
}

function EventRow({ item, onPress }) {
  const scoreColor = item.status === 'GRANTED' ? COLORS.green : item.status === 'DENIED' ? COLORS.red : COLORS.amber;
  return (
    <TouchableOpacity style={styles.eventRow} onPress={onPress} activeOpacity={0.85}>
      <View style={[styles.eventDot, { backgroundColor: item.dot }]} />
      <View style={styles.eventAvatar}>
        <Text style={styles.eventAvatarText}>👤</Text>
      </View>
      <View style={styles.eventMain}>
        <Text style={styles.eventName}>{item.name}</Text>
        <Text style={styles.eventSub}>{item.sub}</Text>
        <StatusBadge status={item.status} />
      </View>
      <View style={styles.eventRight}>
        <Text style={[styles.eventScore, { color: scoreColor }]}>{item.score}</Text>
        <Text style={styles.eventTime}>{item.time}</Text>
        <Text style={styles.eventId}>{item.id}</Text>
      </View>
    </TouchableOpacity>
  );
}

export default function AccessHistory({ navigation }) {
  const [search, setSearch] = useState('');
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
          <Text style={styles.headerTime}>Admin</Text>
          <View style={styles.avatarCircle}><Text>👤</Text></View>
        </View>
      </View>
      {/* Dropdown */}
      <View style={styles.dropdown}>
        <Text style={styles.dropdownText}>Access History (Detailed Log)</Text>
        <Text style={styles.dropdownArrow}>▼</Text>
      </View>

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false}>
        {/* Page title */}
        <View style={styles.pageTitleRow}>
          <View>
            <Text style={styles.pageTitle}>ACCESS HISTORY</Text>
            <Text style={styles.pageSubtitle}>Detailed biometric access event log</Text>
          </View>
          <View style={styles.titleBtns}>
            <TouchableOpacity style={styles.exportBtn}>
              <Text style={styles.exportBtnText}>↓ EXPORT</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.shareBtn}>
              <Text style={styles.shareBtnText}>↗ SHARE</Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* Stats row */}
        <View style={styles.statsRow}>
          <View style={[styles.statCard, { borderColor: COLORS.green }]}>
            <Text style={[styles.statNum, { color: COLORS.green }]}>247</Text>
            <Text style={styles.statLabel}>GRANTED</Text>
            <Text style={styles.statIcon}>✓</Text>
          </View>
          <View style={[styles.statCard, { borderColor: COLORS.red }]}>
            <Text style={[styles.statNum, { color: COLORS.red }]}>18</Text>
            <Text style={styles.statLabel}>DENIED</Text>
            <Text style={styles.statIcon}>✗</Text>
          </View>
          <View style={[styles.statCard, { borderColor: COLORS.amber }]}>
            <Text style={[styles.statNum, { color: COLORS.amber }]}>5</Text>
            <Text style={styles.statLabel}>ERRORS</Text>
            <Text style={styles.statIcon}>⚠</Text>
          </View>
        </View>

        {/* Filters card */}
        <View style={styles.card}>
          <Text style={styles.filterTitle}>FILTERS & SEARCH</Text>
          <View style={styles.filtersGrid}>
            <View style={styles.filterHalf}>
              <Text style={styles.filterLabel}>DATE RANGE</Text>
              <View style={styles.filterSelect}>
                <Text style={styles.filterSelectText}>Last 24 Hours</Text>
                <Text style={styles.filterArrow}>▼</Text>
              </View>
            </View>
            <View style={styles.filterHalf}>
              <Text style={styles.filterLabel}>DEVICE</Text>
              <View style={styles.filterSelect}>
                <Text style={styles.filterSelectText}>All Devices</Text>
                <Text style={styles.filterArrow}>▼</Text>
              </View>
            </View>
            <View style={styles.filterHalf}>
              <Text style={styles.filterLabel}>RESULT</Text>
              <View style={styles.filterSelect}>
                <Text style={styles.filterSelectText}>All Results</Text>
                <Text style={styles.filterArrow}>▼</Text>
              </View>
            </View>
            <View style={styles.filterHalf}>
              <Text style={styles.filterLabel}>USER</Text>
              <View style={styles.filterSelect}>
                <Text style={styles.filterSelectText}>All Users</Text>
                <Text style={styles.filterArrow}>▼</Text>
              </View>
            </View>
          </View>
          <View style={styles.searchRow}>
            <Text style={styles.searchIcon}>🔍</Text>
            <TextInput
              style={styles.searchInput}
              placeholder="Search events..."
              placeholderTextColor={COLORS.textDim}
              value={search}
              onChangeText={setSearch}
            />
          </View>
        </View>

        {/* Event log */}
        <View style={styles.card}>
          <Text style={styles.filterTitle}>EVENT LOG</Text>
          {events.map((item) => (
            <EventRow
              key={item.id}
              item={item}
              onPress={() => navigation?.navigate('AccessEvent', { event: item })}
            />
          ))}
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
    backgroundColor: COLORS.headerBg, paddingHorizontal: 16, paddingTop: 44, paddingBottom: 10,
    borderBottomWidth: 1, borderBottomColor: COLORS.cardBorder,
  },
  headerLeft: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  logoVein: { color: '#fff', fontWeight: '900', fontSize: 18, letterSpacing: 1 },
  logoGuard: { color: COLORS.teal, fontWeight: '900', fontSize: 18, letterSpacing: 1, marginRight: 10 },
  mqttBadge: { flexDirection: 'row', alignItems: 'center', borderWidth: 1, borderColor: COLORS.green, borderRadius: 20, paddingHorizontal: 8, paddingVertical: 3 },
  mqttDot: { width: 7, height: 7, borderRadius: 3.5, backgroundColor: COLORS.green, marginRight: 4 },
  mqttText: { color: COLORS.green, fontSize: 9, fontWeight: '700', letterSpacing: 0.5 },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  headerTime: { color: COLORS.text, fontSize: 12 },
  avatarCircle: { width: 34, height: 34, borderRadius: 17, backgroundColor: COLORS.cardBorder, alignItems: 'center', justifyContent: 'center' },
  dropdown: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    marginHorizontal: 14, marginVertical: 8, padding: 10,
    backgroundColor: COLORS.cardBg, borderRadius: 8, borderWidth: 1, borderColor: COLORS.cardBorder,
  },
  dropdownText: { color: COLORS.teal, fontSize: 12 },
  dropdownArrow: { color: COLORS.textDim, fontSize: 10 },
  scroll: { flex: 1, paddingHorizontal: 14 },
  pageTitleRow: { flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 12, marginTop: 4 },
  pageTitle: { color: COLORS.white, fontSize: 22, fontWeight: '900', letterSpacing: 2 },
  pageSubtitle: { color: COLORS.textDim, fontSize: 11, marginTop: 2 },
  titleBtns: { flexDirection: 'row', gap: 8, marginTop: 4 },
  exportBtn: { backgroundColor: COLORS.cardBg, borderWidth: 1, borderColor: COLORS.teal, borderRadius: 6, paddingHorizontal: 12, paddingVertical: 7 },
  exportBtnText: { color: COLORS.teal, fontSize: 11, fontWeight: '700' },
  shareBtn: { backgroundColor: COLORS.cardBg, borderWidth: 1, borderColor: '#c000ff', borderRadius: 6, paddingHorizontal: 12, paddingVertical: 7 },
  shareBtnText: { color: '#c000ff', fontSize: 11, fontWeight: '700' },
  statsRow: { flexDirection: 'row', gap: 8, marginBottom: 12 },
  statCard: {
    flex: 1, backgroundColor: COLORS.cardBg, borderRadius: 10, borderWidth: 1,
    padding: 12, alignItems: 'flex-start',
  },
  statNum: { fontSize: 26, fontWeight: '900' },
  statLabel: { color: COLORS.textDim, fontSize: 10, letterSpacing: 1, marginTop: 2 },
  statIcon: { alignSelf: 'flex-end', fontSize: 18, marginTop: 4 },
  card: { backgroundColor: COLORS.cardBg, borderRadius: 10, borderWidth: 1, borderColor: COLORS.cardBorder, padding: 14, marginBottom: 12 },
  filterTitle: { color: COLORS.teal, fontSize: 12, fontWeight: '800', letterSpacing: 2, marginBottom: 12 },
  filtersGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginBottom: 10 },
  filterHalf: { width: '48%' },
  filterLabel: { color: COLORS.textDim, fontSize: 10, marginBottom: 5 },
  filterSelect: { flexDirection: 'row', justifyContent: 'space-between', backgroundColor: '#091525', borderWidth: 1, borderColor: COLORS.cardBorder, borderRadius: 6, paddingHorizontal: 10, paddingVertical: 9 },
  filterSelectText: { color: COLORS.text, fontSize: 12 },
  filterArrow: { color: COLORS.textDim, fontSize: 9 },
  searchRow: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#091525', borderWidth: 1, borderColor: COLORS.cardBorder, borderRadius: 6, paddingHorizontal: 10 },
  searchIcon: { fontSize: 14, marginRight: 8 },
  searchInput: { flex: 1, color: COLORS.white, paddingVertical: 10, fontSize: 13 },
  eventRow: {
    flexDirection: 'row', alignItems: 'center',
    paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: COLORS.cardBorder,
  },
  eventDot: { width: 9, height: 9, borderRadius: 4.5, marginRight: 8 },
  eventAvatar: { width: 40, height: 40, borderRadius: 20, backgroundColor: '#1a2e4a', alignItems: 'center', justifyContent: 'center', marginRight: 10 },
  eventAvatarText: { fontSize: 20 },
  eventMain: { flex: 1 },
  eventName: { color: COLORS.white, fontSize: 13, fontWeight: '700' },
  eventSub: { color: COLORS.textDim, fontSize: 10, marginBottom: 4 },
  badge: { alignSelf: 'flex-start', paddingHorizontal: 8, paddingVertical: 2, borderRadius: 4, borderWidth: 1 },
  badgeText: { fontSize: 9, fontWeight: '800', letterSpacing: 1 },
  eventRight: { alignItems: 'flex-end' },
  eventScore: { fontSize: 13, fontWeight: '800' },
  eventTime: { color: COLORS.text, fontSize: 11, marginTop: 2 },
  eventId: { color: COLORS.textDim, fontSize: 10, marginTop: 2 },
});
