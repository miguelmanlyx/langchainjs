import { ChatAIBadgr } from "@langchain/community/chat_models/aibadgr";
import { HumanMessage } from "@langchain/core/messages";

const model = new ChatAIBadgr({
  temperature: 0.9,
  // In Node.js defaults to process.env.AIBADGR_API_KEY
  apiKey: "YOUR-API-KEY",
});

console.log(await model.invoke([new HumanMessage("Hello there!")]));
