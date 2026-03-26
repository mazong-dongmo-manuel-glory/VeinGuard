import React from 'react';
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

function ActionButton({ icon, title, color, bg }) {
  return (
    <TouchableOpacity style={[styles.actionBtn, { borderColor: `${color}40`, backgroundColor: `${color}10` }]}>
      <Ionicons name={icon} size={20} color={color} style={{ marginBottom: 6 }} />
      <Text style={[styles.actionBtnText, { color }]}>{title.toUpperCase()}</Text>
    </TouchableOpacity>
  );
}

function TimelineItem({ icon, title, desc, meta, time, dotColor, isLast = false }) {
  return (
    <View style={styles.timelineItem}>
      <View style={styles.timelineLeft}>
        <View style={[styles.timelineDot, { backgroundColor: dotColor, shadowColor: dotColor }]} />
        {!isLast && <View style={styles.timelineLine} />}
      </View>
      <View style={styles.timelineContent}>
        <View style={styles.timelineHeader}>
          <Text style={styles.timelineTitle}>{title.toUpperCase()}</Text>
          <Text style={styles.timelineTime}>{time}</Text>
        </View>
        <Text style={styles.timelineDesc}>{desc}</Text>
        <Text style={[styles.timelineMeta, { color: dotColor }]}>{meta}</Text>
      </View>
    </View>
  );
}

function TelemetryRow({ label, value, valueColor = COLORS.white }) {
  return (
    <View style={styles.telemetryRow}>
      <Text style={styles.telemetryLabel}>{label.toUpperCase()}</Text>
      <Text style={[styles.telemetryValue, { color: valueColor }]}>{value}</Text>
    </View>
  );
}

function CodeBlock({ title, topic, payload }) {
  return (
    <BlurView intensity={5} style={styles.codeCard}>
      <Text style={styles.codeTitle}>{title.toUpperCase()}</Text>
      <Text style={styles.codeTopic}>{topic}</Text>
      <View style={styles.codeBox}>
        <Text style={styles.codeText}>{payload}</Text>
      </View>
    </BlurView>
  );
}

export default function AccessEvent({ navigation, route }) {
  const { t } = useTranslation();
  const event = route?.params?.event;
  const eventId = event?.id || 'EVT-8743';
  const eventName = event?.name || 'John Mitchell';
  const eventTime = event?.time || '14:23:45';
  const eventStatus = event?.status || 'GRANTED';
  const eventScore = event?.score || '98.7%';
  const userSlug = eventName.toLowerCase().replace(/\s+/g, '.');

  const statusColor =
    eventStatus === 'GRANTED'
      ? COLORS.neonGreen
      : eventStatus === 'DENIED'
        ? COLORS.neonRed
        : COLORS.neonAmber;

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" />
      <LinearGradient colors={GRADIENTS.primary} style={StyleSheet.absoluteFill} />

      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation?.goBack()} style={styles.backBtn}>
          <Ionicons name="chevron-back" size={24} color={COLORS.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>EVENT LOG</Text>
        <View style={styles.spacer} />
      </View>

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        <View style={styles.titleSection}>
          <Text style={styles.pageTitle}>{eventId}</Text>
          <View style={[styles.statusBadge, { borderColor: statusColor, backgroundColor: `${statusColor}15` }]}>
            <Text style={[styles.statusText, { color: statusColor }]}>{eventStatus}</Text>
          </View>
        </View>

        <View style={styles.actionsRow}>
          <ActionButton icon="refresh" title="RE-RUN" color={COLORS.neonCyan} />
          <ActionButton icon="flag" title="FLAG" color={COLORS.neonRed} />
          <ActionButton icon="copy" title="COPY" color={COLORS.neonPurple} />
        </View>

        <BlurView intensity={15} tint="dark" style={styles.card}>
          <Text style={styles.cardTitle}>EVENT TIMELINE</Text>
          <View style={styles.timelineContainer}>
            <TimelineItem
              icon="play"
              title="Scan Start"
              desc="Biometric sensor activated, multimodal capture initiated"
              meta="Duration: 280ms"
              time="14:23:42.341"
              dotColor={COLORS.neonCyan}
            />
            <TimelineItem
              icon="cloud-upload"
              title="Publish"
              desc="Biometric summary transmitted via MQTT to authentication gateway"
              meta="Latency: 127ms"
              time="14:23:44.188"
              dotColor={COLORS.neonPurple}
            />
            <TimelineItem
              icon="git-commit"
              title="Decision"
              desc="AI pattern matching completed, confidence score calculated"
              meta={`Processing: 768ms · Confidence: ${eventScore}`}
              time="14:23:44.956"
              dotColor={statusColor}
            />
            <TimelineItem
              icon="power"
              title="Relay Trigger"
              desc="Access granted, door relay activated for 5 seconds"
              meta="Response: 67ms · Status: SUCCESS"
              time="14:23:45.021"
              dotColor={COLORS.neonGreen}
              isLast
            />
          </View>
        </BlurView>

        <BlurView intensity={10} style={styles.card}>
          <Text style={styles.cardTitle}>RPI TELEMETRY</Text>
          <TelemetryRow label="Device ID" value="BG-RPI-01" />
          <TelemetryRow label="RSSI" value="-42 dBm" valueColor={COLORS.neonGreen} />
          <TelemetryRow label="Uptime" value="72h 14m" />
          <TelemetryRow label="Internal Temp" value="34.2°C" valueColor={COLORS.neonAmber} />
        </BlurView>

        <View style={styles.logSection}>
          <Text style={styles.sectionTitle}>RAW PAYLOADS</Text>
          <CodeBlock
            title="Scan Request"
            topic="bioguard/cmd/access/scan"
            payload={`{\n  "eventId": "${eventId}",\n  "deviceId": "BG-RPI-01",\n  "timestamp": "2024-01-15T14:23:42.341Z",\n  "userId": "${userSlug}",\n  "sensorData": {\n    "quality": 0.94,\n    "modalities": ["palmprint", "finger_geometry"]\n  }\n}`}
          />
          <CodeBlock
            title="Auth Response"
            topic="bioguard/res/access/scan/mobile-demo"
            payload={`{\n  "eventId": "${eventId}",\n  "result": "${eventStatus}",\n  "confidence": ${eventScore === '--' ? 'null' : (Number(eventScore.replace('%', '')) / 100).toFixed(3)},\n  "userId": "${userSlug}",\n  "timestamp": "2024-01-15T14:23:44.956Z",\n  "doorAction": "UNLOCK_5S"\n}`}
          />
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
  spacer: { width: 40 },

  scroll: { flex: 1, paddingHorizontal: 20 },
  scrollContent: { paddingTop: 10 },

  titleSection: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 25 },
  pageTitle: { color: COLORS.white, fontSize: 32, fontWeight: '900', letterSpacing: 1 },
  statusBadge: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 10, borderWidth: 1 },
  statusText: { fontSize: 11, fontWeight: '900', letterSpacing: 1 },

  actionsRow: { flexDirection: 'row', gap: 10, marginBottom: 25 },
  actionBtn: { flex: 1, borderRadius: 15, borderWidth: 1, paddingVertical: 15, alignItems: 'center', shadowOpacity: 0.1, shadowRadius: 10 },
  actionBtnText: { fontSize: 10, fontWeight: '900', letterSpacing: 1 },

  card: { borderRadius: 24, padding: 20, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.05)', marginBottom: 20, overflow: 'hidden' },
  cardTitle: { color: COLORS.textDim, fontSize: 10, fontWeight: '900', letterSpacing: 2, marginBottom: 20 },

  timelineContainer: { paddingLeft: 5 },
  timelineItem: { flexDirection: 'row', gap: 20, marginBottom: 0 },
  timelineLeft: { alignItems: 'center', width: 10 },
  timelineDot: { width: 10, height: 10, borderRadius: 5, zIndex: 1, shadowOpacity: 0.8, shadowRadius: 4 },
  timelineLine: { width: 2, flex: 1, backgroundColor: 'rgba(255, 255, 255, 0.05)', marginVertical: 2 },
  timelineContent: { flex: 1, paddingBottom: 25 },
  timelineHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 5 },
  timelineTitle: { color: COLORS.white, fontSize: 13, fontWeight: '800', letterSpacing: 0.5 },
  timelineTime: { color: COLORS.textDim, fontSize: 10, fontWeight: '600' },
  timelineDesc: { color: COLORS.textSecondary, fontSize: 12, lineHeight: 18, marginBottom: 5 },
  timelineMeta: { fontSize: 10, fontWeight: '700' },

  telemetryRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: 'rgba(255, 255, 255, 0.03)' },
  telemetryLabel: { color: COLORS.textSecondary, fontSize: 11, fontWeight: '700' },
  telemetryValue: { fontSize: 13, fontWeight: '800' },

  logSection: { gap: 12 },
  sectionTitle: { color: COLORS.textDim, fontSize: 10, fontWeight: '900', letterSpacing: 2, marginBottom: 15 },
  codeCard: { borderRadius: 20, padding: 15, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.05)', marginBottom: 15, overflow: 'hidden' },
  codeTitle: { color: COLORS.neonCyan, fontSize: 11, fontWeight: '900', letterSpacing: 1, marginBottom: 5 },
  codeTopic: { color: COLORS.textDim, fontSize: 10, marginBottom: 10 },
  codeBox: { backgroundColor: 'rgba(0, 0, 0, 0.3)', borderRadius: 12, padding: 12, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.05)' },
  codeText: { color: '#8899aa', fontSize: 11, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace', lineHeight: 16 },
});
