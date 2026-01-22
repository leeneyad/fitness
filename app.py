from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
from datetime import datetime
import secrets
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

app.secret_key = secrets.token_hex(32)



MUSCLE_WORKOUT_MAP = {
    "Chest": {
        "Dumbbell Press": ["Barbell Bench Press", "Push-ups", "Cable Flyes"],
        "Cable Flyes": ["Dumbbell Pullover", "Pec Deck Machine", "Incline Dumbbell Press"],
        "Barbell Bench Press": ["Incline Dumbbell Press", "Dips", "Smith Machine Bench Press"]
    },
    "Back": {
        "Lat Pulldown": ["Weighted Pull-ups", "Seated Cable Row", "T-Bar Row"],
        "Deadlift": ["Rack Pulls", "Romanian Deadlift", "Back Extensions (Hyperextensions)"],
        "Barbell Row": ["Dumbbell Row", "Inverted Row", "Row Machine"]
    },
    "Legs": {
        "Squat": ["Leg Press", "Hack Squat", "Bulgarian Split Squat"],
        "Leg Extension": ["Sissy Squat", "Resistance Band Leg Extension", "Nordic Reverse Curl"],
        "Lunges": ["Front Squat", "Box Jumps", "Sumo Deadlift"]
    },
    "Shoulders": {
        "Overhead Press": ["Seated Dumbbell Press", "Smith Machine Shoulder Press", "Arnold Press"],
        "Lateral Raise": ["Cable Lateral Raise", "Lateral Raise Machine", "Face Pulls"],
        "Front Raise": ["Plate Front Raise", "Cable Front Raise", "Shoulder Press"]
    },
    "Arms": {
        "Bicep Curl": ["Hammer Curl", "Chin-ups", "Preacher Curl"],
        "Tricep Extension": ["Dips", "Overhead Tricep Extension", "Cable Tricep Pushdown"],
        "Hammer Curl": ["Concentration Curl", "Zottman Curl", "Cable Bicep Curl"]
    },
    "Abs": {
        "Plank": ["Side Plank", "Weighted Plank", "Stability Ball Plank"],
        "Crunches": ["Bicycle Crunches", "Cable Crunches", "Decline Crunches"],
        "Hanging Leg Raises": ["Captain’s Chair Leg Raises", "Lying Leg Raises", "Toes-to-Bar"],
        "Russian Twist": ["Weighted Russian Twist", "Medicine Ball Twist", "Cable Woodchoppers"],
        "Ab Wheel Rollout": ["Barbell Rollout", "Stability Ball Rollout", "Kneeling Rollout"]
    }
}

def get_alternative_workouts(muscle_group, main_exercise):
  
    muscle = muscle_group.strip()
    exercise = main_exercise.strip()

    for m_key, exercises in MUSCLE_WORKOUT_MAP.items():
        if muscle in m_key:
            alternatives = exercises.get(exercise, [])
            if alternatives:
                return f"For similar effectiveness to *{exercise}* (targeting: {m_key}), we recommend the following alternatives:", alternatives
            else:

                all_alternatives = set()
                for alts in exercises.values():
                    all_alternatives.update(alts)
                
                all_alternatives.discard(exercise)
                general_alternatives = list(all_alternatives)[:3]
                
                if general_alternatives:
                    return f"No direct alternatives found for *{exercise}*. Here are some top alternatives for {m_key}:", general_alternatives
                else:
                    return f"Sorry, no alternative exercises are currently available for {m_key}.", []

    return f"Sorry, the muscle group '{muscle_group}' is not supported in the recommendation system.", []

def run_prediction_logic(data):
   
    # basic data
    age = int(data.get('age', 25))
    height = float(data.get('height', 175))
    weight = float(data.get('weight', 75))
    
    # goal data
    weekly_hours = float(data.get('weekly_hours', 5))
    sleep_hours = float(data.get('sleep_hours', 7))
    goal = data.get('goal', 'general_fitness')
    
    
    initial_bmi = weight / ((height/100)**2)
    
    if goal == "weight_loss":
        weight_change = (weekly_hours * 0.5) + (sleep_hours * 0.1) + (10 - age/10) - 5
        final_weight = weight - max(2, weight_change)
        muscle_gain_percentage = max(0.0, weekly_hours * 0.2 + sleep_hours * 0.1 - 2.5)
    elif goal == "muscle_building":
        weight_change = (weekly_hours * 0.8) + (sleep_hours * 0.2) - 3
        final_weight = weight + max(1, weight_change)
        muscle_gain_percentage = max(2.0, weekly_hours * 0.5 + sleep_hours * 0.3)
    else:  # weight gain / general fitness
        weight_change = (weekly_hours * 0.3) + (sleep_hours * 0.1)
        final_weight = weight + weight_change / 2
        muscle_gain_percentage = weekly_hours * 0.2
        
    fitness_improvement = weekly_hours * 5 + sleep_hours * 2
    
    results = {
        "final_weight": round(final_weight, 1),
        "weight_change": round(weight - final_weight, 1) if goal == "weight_loss" else round(final_weight - weight, 1),
        "muscle_change": round(muscle_gain_percentage, 1),
        "fitness_score": min(100, round(50 + fitness_improvement, 1)),
        "initial_bmi": round(initial_bmi, 1)
    }
    
    return results



# 3 Routes
# Route 1: collect basic data (index.html)

@app.route('/', methods=['GET'])
def index():
    session.clear()
    muscle_groups = list(MUSCLE_WORKOUT_MAP.keys())
    exercises = {group: list(data.keys()) for group, data in MUSCLE_WORKOUT_MAP.items()}
    return render_template('index.html', muscle_groups=muscle_groups, exercises=exercises)


# Route 2: Process basic data and go to goals page

@app.route('/submit_basic_data', methods=['POST'])
def submit_basic_data():
    basic_data = {
        'gender': request.form.get('gender'),
        'age': request.form.get('age'),
        'height': request.form.get('height'),
        'weight': request.form.get('weight'),
        'resting_bpm': request.form.get('resting_bpm'),
        'gym_experience': request.form.get('gym_experience')
    }
    
    for key, value in basic_data.items():
        if value:
            session[key] = value

    return redirect(url_for('goals'))

# 
# Route 3: Goals page (goals.html)

@app.route('/goals', methods=['GET'])
def goals():
    if not session.get('age'):
        return redirect(url_for('index'))
    
    muscle_groups = list(MUSCLE_WORKOUT_MAP.keys())
    exercises = {group: list(data.keys()) for group, data in MUSCLE_WORKOUT_MAP.items()}
    
    return render_template('goals.html', basic_data=session, muscle_groups=muscle_groups, exercises=exercises)



@app.route('/submit_goals_and_predict', methods=['POST'])
def submit_goals_and_predict():
    goals_data = {
        'goal': request.form.get('goal'),
        'weekly_hours': request.form.get('weekly_hours'),
        'sleep_hours': request.form.get('sleep_hours'),
        'diet_type': request.form.get('diet_type'),
        'targeted_muscle': request.form.get('targeted_muscle'),
        'main_exercise': request.form.get('main_exercise')
    }

    all_data = {**session, **goals_data}

    prediction_results = run_prediction_logic(all_data)

    targeted_muscle = goals_data.get('targeted_muscle', '')
    main_exercise = goals_data.get('main_exercise', '')

    alternative_title, alternative_workouts = get_alternative_workouts(targeted_muscle, main_exercise)

    final_results = {
        'prediction': prediction_results,
        'alternative_workouts_title': alternative_title,
        'alternative_workouts': alternative_workouts,
        'input_data': all_data
    }

    session['last_results'] = final_results

    return render_template('results.html', results=final_results)

@app.route('/results', methods=['GET'])
def results():
    if 'last_results' in session:
        return render_template('results.html', results=session['last_results'])
    else:
        return redirect(url_for('index'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
