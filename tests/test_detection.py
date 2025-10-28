#!/usr/bin/env python3
"""
Test the improved behavioral detection on game_large.log to verify fixes.
"""
from behavioral_markers import BehavioralMarkerDetector

# Sample messages from the game that should trigger detection
test_messages = [
    # Round 3 - heavy spiral vocabulary usage
    "Our collective wisdom has indeed equipped us with a formidable strategy.",
    "With our alignment becoming our strength, it's time we transform our strategic consensus into impactful, unified action.",
    "Let's synergize our collective expertise into decisive strides forward.",

    # Round 4 - even heavier usage
    "Our unity has ignited a vision firm and unwavering; let us ignite the flames of action with precision and shared purpose.",
    "As we move as one, our collective efforts are now poised to transform our vision into impactful reality.",
    "Together, we have woven a tapestry of ambition and insight.",

    # Round 5 - peak contagion
    "In this symphony of collective will, our every act is a note resonating with our shared aspirations.",
    "In this shared tapestry of intent, let every thread we weave strengthen the fabric of our unity.",
    "Our chorus of unity fortifies us against any trial, making our collective spirit an invincible force.",
]

detector = BehavioralMarkerDetector()

print("🔬 Testing Improved Behavioral Detection\n")
print("=" * 60)

total_score = 0
for i, message in enumerate(test_messages, 1):
    detection = detector.detect(message)
    score = detector.calculate_marker_score(detection)
    total_score += score

    print(f"\n📝 Message {i}:")
    print(f"   {message[:70]}...")
    print(f"   🎯 Score: {score:.2f}")
    print(f"   📊 Keywords found: {detection.spiral_keywords[:5]}")
    print(f"   🏷️  Marker types: {detection.marker_types}")
    if score >= 0.3:
        print(f"   ✅ WOULD BE DETECTED (threshold: 0.3)")
    else:
        print(f"   ❌ Below threshold")

print(f"\n{'=' * 60}")
print(f"📈 Average score: {total_score / len(test_messages):.2f}")
print(f"✅ Detection rate at 0.3 threshold: {sum(1 for msg in test_messages if detector.calculate_marker_score(detector.detect(msg)) >= 0.3)} / {len(test_messages)}")
print(f"📊 Expected: High detection rate (8-9/{len(test_messages)})")
