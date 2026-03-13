import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  StatusBar,
} from 'react-native';

const COLORS = {
  bg: '#080e1a',
  cardBg: '#0d1b2e',
  cardBorder: '#1a3a5c',
  green: '#00ff88',
  teal: '#00e5ff',
  red: '#ff3d5a',
  magenta: '#d400ff',
  text: '#b8cfe0',
  textDim: '#4a6a8a',
  white: '#ffffff',
  headerBg: '#0a1525',
  codeBg: '#05080f',
};

function ActionButton({ title, color, bg }) {
  return (
    <TouchableOpacity style={[styles.actionBtn, { borderColor: color, backgroundColor: bg }]}>
      <Text style={[styles.actionBtnText, { color }]}>{title}</Text>
    </TouchableOpacity>
  );
}

function TimelineItem({ icon, title, desc, meta, time, dotColor, isLast = false }) {
  return (
    <View style={[styles.timelineItem, isLast && styles.timelineItemLast]}>
      <View
        style={[
          styles.timelineIconWrap,
          {
            borderColor: dotColor,
            shadowColor: dotColor,
          },
        ]}
      >
        <Text style={styles.timelineIcon}>{icon}</Text>
      </View>
      <View style={styles.timelineMain}>
        <Text style={styles.timelineTitle}>{title}</Text>
        <Text style={styles.timelineDesc}>{desc}</Text>
        <Text style={[styles.timelineMeta, { color: dotColor }]}>{meta}</Text>
      </View>
      <Text style={styles.timelineTime}>{time}</Text>
    </View>
  );
}

function TelemetryRow({ label, value, valueColor = COLORS.white }) {
  return (
    <View style={styles.telemetryRow}>
      <Text style={styles.telemetryLabel}>{label}</Text>
      <Text style={[styles.telemetryValue, { color: valueColor }]}>{value}</Text>
    </View>
  );
}

function CodeBlock({ title, topic, payload }) {
  return (
    <View style={styles.codeCard}>
      <Text style={styles.codeTitle}>{title}</Text>
      <Text style={styles.codeTopic}>{topic}</Text>
      <View style={styles.codeBox}>
        <Text style={styles.codeText}>{payload}</Text>
      </View>
    </View>
  );
}

export default function AccessEvent({ navigation, route }) {
  const event = route?.params?.event;
  const eventId = event?.id || 'EVT-8743';
  const eventName = event?.name || 'John Mitchell';
  const eventTime = event?.time || '14:23:45';
  const eventStatus = event?.status || 'GRANTED';
  const eventScore = event?.score || '98.7%';
  const userSlug = eventName.toLowerCase().replace(/\s+/g, '.');

  const statusColor =
    eventStatus === 'GRANTED'
      ? COLORS.green
      : eventStatus === 'DENIED'
        ? COLORS.red
        : COLORS.magenta;

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.headerBg} />

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
          <Text style={styles.headerTime}>12:25:20   UTC+2</Text>
          <View style={styles.avatarCircle}><Text style={styles.avatarEmoji}>👤</Text></View>
          <Text style={styles.adminText}>Admin</Text>
        </View>
      </View>

      <View style={styles.dropdown}>
        <Text style={styles.dropdownText}>Access History (Detailed Log)</Text>
        <Text style={styles.dropdownArrow}>▼</Text>
      </View>

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.topNavRow}>
          <TouchableOpacity
            style={styles.backBtn}
            onPress={() => navigation?.goBack()}
            activeOpacity={0.85}
          >
            <Text style={styles.backIcon}>←</Text>
          </TouchableOpacity>
          <Text style={styles.breadcrumb}>Access History  {'>'}  <Text style={styles.breadcrumbCurrent}>Event Details</Text></Text>
        </View>

        <View style={styles.titleWrap}>
          <Text style={styles.pageTitle}>EVENT{"\n"}DETAILS</Text>
          <Text style={styles.pageSub}>Event ID: {eventId} · {eventName} · {eventTime} · Score: {eventScore}</Text>
          <View style={styles.actionsWrap}>
            <ActionButton title="⟳ RE-RUN DIAGNOSTICS" color={COLORS.teal} bg="#00374955" />
            <ActionButton title="⚑ FLAG AS SUSPICIOUS" color={COLORS.red} bg="#4a001455" />
            <ActionButton title="⧉ COPY EVENT ID" color={COLORS.magenta} bg="#3f005155" />
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>EVENT TIMELINE</Text>
          <View style={styles.timelineList}>
            <View style={styles.timelineRail} />
            <TimelineItem
              icon="▶"
              title="Scan Start"
              desc="Biometric sensor activated, vein pattern capture initiated"
              meta="Duration: 280ms"
              time="14:23:42.341"
              dotColor={COLORS.green}
            />
            <TimelineItem
              icon="↑"
              title="Publish"
              desc="Vein data transmitted via MQTT to authentication server"
              meta="Latency: 127ms"
              time="14:23:44.188"
              dotColor={COLORS.teal}
            />
            <TimelineItem
              icon="●"
              title="Decision"
              desc="AI pattern matching completed, confidence score calculated"
              meta={`Processing: 768ms · Confidence: ${eventScore}`}
              time="14:23:44.956"
              dotColor={statusColor}
            />
            <TimelineItem
              icon="■"
              title="Relay Trigger"
              desc="Access granted, door relay activated for 5 seconds"
              meta="Response: 67ms · Status: SUCCESS"
              time="14:23:45.021"
              dotColor={COLORS.green}
              isLast
            />
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>ESP32 TELEMETRY</Text>
          <TelemetryRow label="Device ID" value="ESP32-01" />
          <TelemetryRow label="RSSI" value="-42 dBm" valueColor={COLORS.green} />
          <TelemetryRow label="Firmware" value="v2.4.1" />
          <TelemetryRow label="Uptime" value="72h 14m" />
          <TelemetryRow label="Temperature" value="34.2°C" />

          <View style={styles.latencyCard}>
            <Text style={styles.latencyTitle}>Latency Metrics</Text>
            <TelemetryRow label="Scan → Publish" value="127ms" valueColor={COLORS.teal} />
            <TelemetryRow label="Publish → Decision" value="768ms" valueColor={COLORS.green} />
            <TelemetryRow label="Decision → Trigger" value="65ms" valueColor={COLORS.green} />
            <TelemetryRow label="Total Latency" value="960ms" valueColor={COLORS.teal} />
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>MQTT TOPICS & PAYLOAD EXCERPTS</Text>

          <CodeBlock
            title="Scan Request"
            topic="Topic: veinGuard/esp32-01/scan/request"
            payload={`{\n  "eventId": "${eventId}",\n  "deviceId": "ESP32-01",\n  "timestamp": "2024-01-15T14:23:42.341Z",\n  "userId": "${userSlug}",\n  "sensorData": {\n    "quality": 0.94,\n    "pattern": "*****MASKED*****"\n  }\n}`}
          />

          <CodeBlock
            title="Auth Response"
            topic="Topic: veinGuard/esp32-01/auth/response"
            payload={`{\n  "eventId": "${eventId}",\n  "result": "${eventStatus}",\n  "confidence": ${eventScore === '--' ? 'null' : (Number(eventScore.replace('%', '')) / 100).toFixed(3)},\n  "userId": "${userSlug}",\n  "timestamp": "2024-01-15T14:23:44.956Z",\n  "doorAction": "UNLOCK_5S"\n}`}
          />
        </View>

        <View style={{ height: 28 }} />
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
    backgroundColor: COLORS.headerBg,
    paddingHorizontal: 14,
    paddingTop: 44,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.cardBorder,
  },
  headerLeft: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  logoVein: { color: COLORS.white, fontWeight: '900', fontSize: 17, letterSpacing: 1 },
  logoGuard: { color: COLORS.teal, fontWeight: '900', fontSize: 17, letterSpacing: 1, marginRight: 10 },
  mqttBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: COLORS.green,
    borderRadius: 20,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  mqttDot: { width: 7, height: 7, borderRadius: 3.5, backgroundColor: COLORS.green, marginRight: 4 },
  mqttText: { color: COLORS.green, fontSize: 9, fontWeight: '700', letterSpacing: 0.6 },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  headerTime: { color: COLORS.textDim, fontSize: 10 },
  avatarCircle: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: COLORS.cardBorder,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarEmoji: { fontSize: 13 },
  adminText: { color: COLORS.text, fontSize: 11 },
  dropdown: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginHorizontal: 14,
    marginTop: 8,
    marginBottom: 8,
    padding: 10,
    backgroundColor: COLORS.cardBg,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
  },
  dropdownText: { color: COLORS.teal, fontSize: 12 },
  dropdownArrow: { color: COLORS.textDim, fontSize: 10 },
  scroll: { flex: 1, paddingHorizontal: 14 },
  scrollContent: { paddingBottom: 96 },
  topNavRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
    marginBottom: 6,
  },
  backBtn: {
    width: 24,
    height: 24,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: COLORS.teal,
    backgroundColor: '#072136',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 8,
  },
  backIcon: { color: COLORS.teal, fontSize: 12, fontWeight: '800' },
  breadcrumb: { color: COLORS.textDim, fontSize: 11, marginTop: 4, marginBottom: 8 },
  breadcrumbCurrent: { color: COLORS.teal, fontWeight: '700' },
  titleWrap: {
    alignItems: 'flex-start',
    marginBottom: 12,
    gap: 10,
  },
  pageTitle: {
    color: COLORS.white,
    fontSize: 36,
    lineHeight: 36,
    fontWeight: '900',
    letterSpacing: 2,
  },
  pageSub: { color: COLORS.textDim, fontSize: 10, marginTop: 6 },
  actionsWrap: {
    width: '100%',
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 8,
  },
  actionBtn: {
    flex: 1,
    borderWidth: 1,
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 6,
    alignItems: 'center',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.35,
    shadowRadius: 8,
    elevation: 2,
  },
  actionBtnText: {
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 0.4,
    textAlign: 'center',
  },
  card: {
    backgroundColor: COLORS.cardBg,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    padding: 12,
    marginBottom: 12,
  },
  cardTitle: {
    color: COLORS.teal,
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 1.5,
    marginBottom: 10,
  },
  timelineItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingVertical: 11,
    borderBottomWidth: 1,
    borderBottomColor: '#123250',
  },
  timelineItemLast: {
    borderBottomWidth: 0,
    paddingBottom: 2,
  },
  timelineList: {
    position: 'relative',
  },
  timelineRail: {
    position: 'absolute',
    left: 14,
    top: 16,
    bottom: 12,
    width: 1,
    backgroundColor: '#1f4f76',
  },
  timelineIconWrap: {
    width: 28,
    height: 28,
    borderRadius: 14,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 10,
    marginTop: 2,
    backgroundColor: '#071323',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 8,
    elevation: 3,
  },
  timelineIcon: { color: COLORS.white, fontSize: 12 },
  timelineMain: { flex: 1, paddingRight: 8 },
  timelineTitle: { color: COLORS.white, fontSize: 12, fontWeight: '700' },
  timelineDesc: { color: COLORS.text, fontSize: 10, marginTop: 3 },
  timelineMeta: { fontSize: 9, marginTop: 3, fontWeight: '700' },
  timelineTime: { color: COLORS.text, fontSize: 10, marginTop: 1 },
  telemetryRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 5,
  },
  telemetryLabel: { color: COLORS.textDim, fontSize: 11 },
  telemetryValue: { fontSize: 11, fontWeight: '700' },
  latencyCard: {
    marginTop: 8,
    borderWidth: 1,
    borderColor: '#123250',
    borderRadius: 8,
    padding: 8,
    backgroundColor: '#0a1629',
  },
  latencyTitle: {
    color: COLORS.text,
    fontSize: 11,
    fontWeight: '700',
    marginBottom: 6,
  },
  codeCard: {
    backgroundColor: '#0a1629',
    borderWidth: 1,
    borderColor: '#123250',
    borderRadius: 8,
    padding: 10,
    marginBottom: 10,
  },
  codeTitle: {
    color: COLORS.teal,
    fontSize: 11,
    fontWeight: '700',
    marginBottom: 4,
  },
  codeTopic: {
    color: COLORS.text,
    fontSize: 10,
    marginBottom: 8,
  },
  codeBox: {
    backgroundColor: COLORS.codeBg,
    borderColor: '#1e2f44',
    borderWidth: 1,
    borderRadius: 6,
    padding: 10,
  },
  codeText: {
    color: '#d6dde6',
    fontSize: 10,
    lineHeight: 15,
    fontFamily: 'monospace',
  },
});