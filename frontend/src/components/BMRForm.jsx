// BMRForm.jsx
import { useState } from 'react';
import api from '@/api';

function BMRForm({ onSessionCreated }) {
  const [weight, setWeight] = useState('');
  const [height, setHeight] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('male');
  const [activity, setActivity] = useState('1.2'); // Default: Sedentary

  async function handleSubmit(e) {
    e.preventDefault();
    const formData = {
      weight: parseFloat(weight),
      height: parseFloat(height),
      age: parseInt(age),
      gender,
      activity: parseFloat(activity)
    };

    try {
      const session_id = await api.createSession(formData);
      // Store session_id in localStorage
      localStorage.setItem('session_id', session_id);

      // Calculate BMR using the Mifflin-St Jeor equation
      let bmr;
      if (gender === 'male') {
        bmr = (10 * formData.weight) + (6.25 * formData.height) - (5 * formData.age) + 5;
      } else {
        bmr = (10 * formData.weight) + (6.25 * formData.height) - (5 * formData.age) - 161;
      }
      const tdee = Math.round(bmr * formData.activity);
      const sessionData = { ...formData, bmr: Math.round(bmr), tdee, session_id };
      onSessionCreated(sessionData);
    } catch (error) {
      console.error("Error creating session:", error);
    }
  }

  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold mb-2">Welcome!</h2>
      <p className="mb-4">
        Let's calculate your Basal Metabolic Rate (BMR) so I can tailor my nutrition advice just for you.
      </p>
      <form onSubmit={handleSubmit} className="bmr-form flex flex-col gap-4">
        <input
          type="number"
          placeholder="Weight (kg)"
          value={weight}
          onChange={(e) => setWeight(e.target.value)}
          className="border p-2 rounded"
        />
        <input
          type="number"
          placeholder="Height (cm)"
          value={height}
          onChange={(e) => setHeight(e.target.value)}
          className="border p-2 rounded"
        />
        <input
          type="number"
          placeholder="Age"
          value={age}
          onChange={(e) => setAge(e.target.value)}
          className="border p-2 rounded"
        />
        <select
          value={gender}
          onChange={(e) => setGender(e.target.value)}
          className="border p-2 rounded"
        >
          <option value="male">Male</option>
          <option value="female">Female</option>
        </select>
        <select
          value={activity}
          onChange={(e) => setActivity(e.target.value)}
          className="border p-2 rounded"
        >
          <option value="1.2">Sedentary (little or no exercise)</option>
          <option value="1.375">Lightly active (light exercise 1-3 days/week)</option>
          <option value="1.55">Moderately active (moderate exercise 3-5 days/week)</option>
          <option value="1.725">Very active (hard exercise 6-7 days/week)</option>
          <option value="1.9">Extra active (very hard exercise or physical job)</option>
        </select>
        <button type="submit" className="bg-primary-blue text-white p-2 rounded">
          Submit
        </button>
      </form>
    </div>
  );
}

export default BMRForm;
