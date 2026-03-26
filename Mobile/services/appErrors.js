export function getAppErrorMessage(t, error, fallbackKey = 'errors.generic') {
  const rawMessage =
    typeof error === 'string'
      ? error
      : error?.message || error?.reason || error?.error || '';

  const message = String(rawMessage || '').trim();

  const mappings = [
    [/^MQTT timeout$/i, 'errors.mqttTimeout'],
    [/^MQTT not connected$/i, 'errors.mqttNotConnected'],
    [/^MQTT client unavailable$/i, 'errors.mqttClientUnavailable'],
    [/^INVALID_CAPTURE$/i, 'errors.invalidCapture'],
    [/^Capture biométrique invalide/i, 'errors.invalidCapture'],
    [/^PROFILE_NOT_FOUND$/i, 'errors.profileNotFound'],
    [/^NO_MATCH_FOUND$/i, 'errors.noMatchFound'],
    [/^BIOMETRIC_MISMATCH$/i, 'errors.biometricMismatch'],
    [/^USER_NOT_FOUND$/i, 'errors.userNotFound'],
    [/^Missing user_id$/i, 'errors.missingUserId'],
    [/^Enrollment failed$/i, 'errors.enrollmentFailed'],
    [/^User update failed$/i, 'errors.userUpdateFailed'],
    [/^User deletion failed$/i, 'errors.userDeleteFailed'],
  ];

  for (const [pattern, key] of mappings) {
    if (pattern.test(message)) {
      return t(key);
    }
  }

  if (!message) {
    return t(fallbackKey);
  }

  return message;
}
