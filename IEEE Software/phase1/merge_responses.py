"""Merge GPT-4 and Claude responses into combined dataset."""
import pandas as pd

# Load both datasets
print("Loading response datasets...")
gpt_df = pd.read_csv('data/responses/test_gpt4_responses.csv')
claude_df = pd.read_csv('data/responses/test_claude_responses.csv')

# Add Claude responses to GPT dataframe
gpt_df['response_claude'] = claude_df['response_claude']

# Statistics
print(f"\nResponse statistics:")
print(f"  GPT-4 responses:   {gpt_df['response_gpt4'].notna().sum()}/{len(gpt_df)}")
print(f"  Claude responses:  {gpt_df['response_claude'].notna().sum()}/{len(gpt_df)}")

# Use Claude as primary, GPT-4 as fallback
gpt_df['response'] = gpt_df['response_claude'].fillna(gpt_df['response_gpt4'])
print(f"  Combined responses: {gpt_df['response'].notna().sum()}/{len(gpt_df)}")

# Save combined dataset
output_file = 'data/responses/test_combined_responses.csv'
gpt_df.to_csv(output_file, index=False)
print(f"\n✅ Saved to: {output_file}")
