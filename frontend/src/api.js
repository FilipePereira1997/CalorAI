// api.js
const BASE_URL = import.meta.env.VITE_API_URL;

async function createSession(userData) {
  const res = await fetch(BASE_URL + '/session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(userData)
  });

  if (!res.ok) {
    return Promise.reject({ status: res.status, data: await res.json() });
  }

  const data = await res.json();
  return data.session_id;
}

async function sendChatMessage(message, sessionData) {
  const payload = {
    text: message,
    user: sessionData
  };
  const res = await fetch(BASE_URL + '/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    return Promise.reject({ status: res.status, data: await res.json() });
  }

  const data = await res.json();
  return data.message;
}

export default {
  createSession,
  sendChatMessage
};
