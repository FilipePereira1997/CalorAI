// Chatbot.jsx
import { useState, useEffect } from 'react';
import { useImmer } from 'use-immer';
import api from '@/api';
import ChatMessages from '@/components/ChatMessages';
import ChatInput from '@/components/ChatInput';
import BMRForm from '@/components/BMRForm';

const BASE_URL = import.meta.env.VITE_API_URL;

function Chatbot() {
  const [messages, setMessages] = useImmer([]);
  const [newMessage, setNewMessage] = useState('');
  // State to store the user's session data (BMR, TDEE, etc.)
  const [bmrData, setBmrData] = useState(null);

  // On initial load, check for an existing session ID in localStorage
  useEffect(() => {
    const storedSessionId = localStorage.getItem('session_id');
    if (storedSessionId) {
      fetch(BASE_URL + '/session/' + storedSessionId)
        .then((res) => res.json())
        .then((data) => {
          if (data && data.data) {
            setBmrData({ ...data.data, session_id: data.session_id });
          }
        })
        .catch((err) => {
          console.error("Error fetching session:", err);
        });
    }
  }, []);

  async function submitNewMessage() {
    const trimmedMessage = newMessage.trim();
    if (!trimmedMessage || (messages.length && messages[messages.length - 1].loading)) return;

    setMessages(draft => {
      draft.push({ role: 'user', content: trimmedMessage });
      draft.push({ role: 'assistant', content: '', loading: true });
    });
    setNewMessage('');

    try {
      // Send the chat message along with the session data to the backend
      const reply = await api.sendChatMessage(trimmedMessage, bmrData);
      setMessages(draft => {
        draft[draft.length - 1].content = reply;
        draft[draft.length - 1].loading = false;
      });
    } catch (err) {
      console.error(err);
      setMessages(draft => {
        draft[draft.length - 1].loading = false;
        draft[draft.length - 1].error = true;
      });
    }
  }

  // Handler to update data (i.e., start a new session)
  function handleUpdateData() {
    localStorage.removeItem('session_id'); // Remove stored session ID
    setBmrData(null); // Reset the session state
    setMessages([]); // Optionally clear existing chat messages
  }

  // If session data is not set, display the BMR form
  if (!bmrData) {
    return (
      <div className="relative grow flex flex-col gap-6 pt-6">
        <BMRForm onSessionCreated={(sessionData) => setBmrData(sessionData)} />
      </div>
    );
  }

  const isLoading = messages.length && messages[messages.length - 1].loading;

  return (
    <div className="relative grow flex flex-col gap-6 pt-6">
      <div className="flex justify-end p-2">
        <button
          onClick={handleUpdateData}
          className="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-1 px-3 rounded"
        >
          Update Data
        </button>
      </div>
      {messages.length === 0 && (
        <div className="mt-3 font-urbanist text-primary-blue text-xl font-light space-y-2">
          <p>👋 Hi there!</p>
          <p>
            Welcome back! Your profile is set with a BMR of {bmrData.bmr} kcal per day and an estimated TDEE of {bmrData.tdee} kcal per day.
            How can I help you with your nutrition goals today?
          </p>
        </div>
      )}
      <ChatMessages messages={messages} isLoading={isLoading} />
      <ChatInput
        newMessage={newMessage}
        isLoading={isLoading}
        setNewMessage={setNewMessage}
        submitNewMessage={submitNewMessage}
      />
    </div>
  );
}

export default Chatbot;
