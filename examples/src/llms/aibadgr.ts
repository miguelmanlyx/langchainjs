import { ChatOpenAI } from "@langchain/openai";

/**
 * Example showing how to use AI Badgr as an OpenAI-compatible endpoint.
 * AI Badgr is a budget/utility provider that implements the OpenAI API.
 *
 * Setup:
 * export AIBADGR_API_KEY="your-api-key"
 */
export const run = async () => {
  // Initialize AI Badgr using the premium tier (recommended for examples)
  const model = new ChatOpenAI({
    apiKey: process.env.AIBADGR_API_KEY,
    model: "premium", // Available tiers: "basic", "normal", "premium"
    temperature: 0.7,
    configuration: {
      baseURL: process.env.AIBADGR_BASE_URL || "https://aibadgr.com/api/v1",
    },
  });

  const res = await model.invoke(
    "What would be a good company name for a company that makes colorful socks?"
  );
  console.log({ res });
};
