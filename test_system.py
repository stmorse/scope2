"""
Test script to verify the MCTS conversation planning system works correctly.
"""

from conversation_planner import ConversationPlanner
from llm_providers import MockProvider
from reward_functions import Agent1LengthReward


def test_basic_functionality():
    """Test basic functionality with mock providers."""
    print("Testing MCTS Conversation Planning System")
    print("=" * 50)
    
    # Create mock providers with predictable responses
    agent1_provider = MockProvider([
        "That's really interesting! I'd love to know more about that topic.",
        "I completely agree with your perspective on this matter.",
        "Could you elaborate on that point? I find it fascinating."
    ])
    
    agent2_provider = MockProvider([
        "I think artificial intelligence will transform how we work.",
        "AI has both exciting opportunities and important challenges.",
        "The key is developing AI responsibly and ethically.",
        "What aspects of AI interest you most?",
        "I believe AI should augment human capabilities."
    ])
    
    # Initialize planner
    planner = ConversationPlanner(
        agent1_provider=agent1_provider,
        agent2_provider=agent2_provider,
        reward_function=Agent1LengthReward(normalize=True, max_length=500),
        max_depth=2,  # Keep shallow for testing
        num_simulations=10,  # Keep low for testing
        exploration_constant=1.414,
        temperature=0.7
    )
    
    # Test conversation planning
    initial_prompt = "What do you think about artificial intelligence?"
    print(f"Initial prompt: '{initial_prompt}'")
    print()
    
    results = planner.plan_conversation(initial_prompt, num_candidates=3)
    
    print(f"\nGenerated {len(results)} results:")
    for i, (candidate, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.4f}")
        print(f"   Response: '{candidate}'")
    
    print(f"\nBest response: '{results[0][0]}'")
    print(f"Best score: {results[0][1]:.4f}")
    
    # Verify results
    assert len(results) > 0, "Should generate at least one result"
    assert all(score >= 0 for _, score in results), "All scores should be non-negative"
    assert results == sorted(results, key=lambda x: x[1], reverse=True), "Results should be sorted by score"
    
    print("\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    test_basic_functionality()
