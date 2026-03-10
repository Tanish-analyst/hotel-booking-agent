## Setup

1. Clone the repo

git clone [https://github.com/Tanish-analyst/hotel-booking-agent.git](https://github.com/Tanish-analyst/hotel-booking-agent.git)

2. Install dependencies

pip install -r requirements.txt

3. Create .env file

Copy .env.example and rename to .env

4. Add your GROQ API key

GROQ_API_KEY=your_key

5. Run Redis

redis-server

6. Run the agent

python hotel_booking_agent.py
