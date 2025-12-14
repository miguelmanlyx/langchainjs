import { describe, it, expect } from "vitest";
import { ChatAIBadgr } from "../chat_models.js";

describe("ChatAIBadgr", () => {
  it("should instantiate with an API key", () => {
    const model = new ChatAIBadgr({
      apiKey: "test-api-key",
      model: "premium",
    });
    expect(model).toBeDefined();
    expect(model._llmType()).toBe("aibadgr");
  });

  it("should throw error if no API key is provided", () => {
    expect(() => {
      new ChatAIBadgr({
        model: "premium",
      });
    }).toThrow(
      'AI Badgr API key not found. Please set the AIBADGR_API_KEY environment variable or pass the key into "apiKey" field.'
    );
  });

  it("should use default model 'premium' when not specified", () => {
    const model = new ChatAIBadgr({
      apiKey: "test-api-key",
    });
    expect(model.model).toBe("premium");
  });

  it("should use custom model when specified", () => {
    const model = new ChatAIBadgr({
      apiKey: "test-api-key",
      model: "basic",
    });
    expect(model.model).toBe("basic");
  });

  it("should have correct lc_secrets", () => {
    const model = new ChatAIBadgr({
      apiKey: "test-api-key",
      model: "premium",
    });
    expect(model.lc_secrets).toEqual({
      apiKey: "AIBADGR_API_KEY",
    });
  });

  it("should have correct lc_namespace", () => {
    const model = new ChatAIBadgr({
      apiKey: "test-api-key",
      model: "premium",
    });
    expect(model.lc_namespace).toEqual(["langchain", "chat_models", "aibadgr"]);
  });

  it("should return profile for known models", () => {
    const model = new ChatAIBadgr({
      apiKey: "test-api-key",
      model: "premium",
    });
    const profile = model.profile;
    expect(profile).toBeDefined();
    expect(profile.maxInputTokens).toBe(32768);
    expect(profile.maxOutputTokens).toBe(16384);
    expect(profile.toolCalling).toBe(true);
  });

  it("should return profile for tier models", () => {
    const basicModel = new ChatAIBadgr({
      apiKey: "test-api-key",
      model: "basic",
    });
    expect(basicModel.profile.maxInputTokens).toBe(8192);

    const normalModel = new ChatAIBadgr({
      apiKey: "test-api-key",
      model: "normal",
    });
    expect(normalModel.profile.maxInputTokens).toBe(16384);

    const premiumModel = new ChatAIBadgr({
      apiKey: "test-api-key",
      model: "premium",
    });
    expect(premiumModel.profile.maxInputTokens).toBe(32768);
  });

  it("should return profile for power-user models", () => {
    const phiModel = new ChatAIBadgr({
      apiKey: "test-api-key",
      model: "phi-3-mini",
    });
    expect(phiModel.profile.maxInputTokens).toBe(8192);

    const mistralModel = new ChatAIBadgr({
      apiKey: "test-api-key",
      model: "mistral-7b",
    });
    expect(mistralModel.profile.maxInputTokens).toBe(16384);

    const llamaModel = new ChatAIBadgr({
      apiKey: "test-api-key",
      model: "llama3-8b-instruct",
    });
    expect(llamaModel.profile.maxInputTokens).toBe(32768);
  });

  it("should return empty profile for unknown models", () => {
    const model = new ChatAIBadgr({
      apiKey: "test-api-key",
      model: "unknown-model",
    });
    const profile = model.profile;
    expect(profile).toEqual({});
  });
});
