import type { ModelProfile } from "@langchain/core/language_models/profile";

// Base profile configuration shared across all models
const BASE_PROFILE: Partial<ModelProfile> = {
  imageInputs: false,
  audioInputs: false,
  pdfInputs: false,
  videoInputs: false,
  reasoningOutput: false,
  imageOutputs: false,
  audioOutputs: false,
  videoOutputs: false,
  toolCalling: true,
  structuredOutput: false,
};

// Helper function to create a profile with tier-specific token limits
const createProfile = (
  maxInputTokens: number,
  maxOutputTokens: number
): ModelProfile => ({
  ...BASE_PROFILE,
  maxInputTokens,
  maxOutputTokens,
});

const PROFILES: Record<string, ModelProfile> = {
  // Tier-based models
  basic: createProfile(8192, 4096),
  normal: createProfile(16384, 8192),
  premium: createProfile(32768, 16384),

  // Power-user model names mapped to tiers
  "phi-3-mini": createProfile(8192, 4096), // Maps to basic
  "mistral-7b": createProfile(16384, 8192), // Maps to normal
  "llama3-8b-instruct": createProfile(32768, 16384), // Maps to premium
};

export default PROFILES;
