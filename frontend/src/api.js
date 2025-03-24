const BASE_URL = import.meta.env.VITE_API_URL;

async function sendChatMessage(message) {
  const res = await fetch(BASE_URL + '/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: message })
  });

  if (!res.ok) {
    return Promise.reject({ status: res.status, data: await res.json() });
  }

  const data = await res.json();
  return data.message;
}

export default {
  sendChatMessage
};
