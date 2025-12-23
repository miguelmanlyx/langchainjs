import type {
  BaseChatModelParams,
  LangSmithParams,
} from "@langchain/core/language_models/chat_models";
import {
  type OpenAIClient,
  type ChatOpenAICallOptions,
  type OpenAIChatInput,
  type OpenAICoreRequestOptions,
  ChatOpenAICompletions,
} from "@langchain/openai";
import { getEnvironmentVariable } from "@langchain/core/utils/env";

type AIBadgrUnsupportedArgs =
  | "frequencyPenalty"
  | "presencePenalty"
  | "logitBias"
  | "functions";

type AIBadgrUnsupportedCallOptions = "functions" | "function_call";

export interface ChatAIBadgrCallOptions
  extends Omit<ChatOpenAICallOptions, AIBadgrUnsupportedCallOptions> {
  response_format?: {
    type: "json_object";
    schema: Record<string, unknown>;
  };
}

export interface ChatAIBadgrInput
  extends Omit<OpenAIChatInput, "openAIApiKey" | AIBadgrUnsupportedArgs>,
    BaseChatModelParams {
  /**
   * AI Badgr API key
   * @default process.env.AIBADGR_API_KEY
   */
  aibadgrApiKey?: string;
  /**
   * API key alias
   * @default process.env.AIBADGR_API_KEY
   */
  apiKey?: string;
}

/**
 * AI Badgr chat model integration.
 *
 * The AI Badgr API is compatible to the OpenAI API with some limitations.
 *
 * Setup:
 * Install `@langchain/community` and set an environment variable named `AIBADGR_API_KEY`.
 *
 * ```bash
 * npm install @langchain/community
 * export AIBADGR_API_KEY="your-api-key"
 * ```
 *
 * ## [Constructor args](https://api.js.langchain.com/classes/_langchain_community.chat_models_aibadgr.ChatAIBadgr.html#constructor)
 *
 * ## [Runtime args](https://api.js.langchain.com/interfaces/_langchain_community.chat_models_aibadgr.ChatAIBadgrCallOptions.html)
 *
 * Runtime args can be passed as the second argument to any of the base runnable methods `.invoke`. `.stream`, `.batch`, etc.
 * They can also be passed via `.withConfig`, or the second arg in `.bindTools`, like shown in the examples below:
 *
 * ```typescript
 * // When calling `.withConfig`, call options should be passed via the first argument
 * const llmWithArgsBound = llm.withConfig({
 *   stop: ["\n"],
 *   tools: [...],
 * });
 *
 * // When calling `.bindTools`, call options should be passed via the second argument
 * const llmWithTools = llm.bindTools(
 *   [...],
 *   {
 *     tool_choice: "auto",
 *   }
 * );
 * ```
 *
 * ## Examples
 *
 * <details open>
 * <summary><strong>Instantiate</strong></summary>
 *
 * ```typescript
 * import { ChatAIBadgr } from '@langchain/community/chat_models/aibadgr';
 *
 * const llm = new ChatAIBadgr({
 *   model: "gpt-4",
 *   temperature: 0,
 *   // other params...
 * });
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Invoking</strong></summary>
 *
 * ```typescript
 * const input = `Translate "I love programming" into French.`;
 *
 * // Models also accept a list of chat messages or a formatted prompt
 * const result = await llm.invoke(input);
 * console.log(result);
 * ```
 *
 * ```txt
 * AIMessage {
 *   "id": "abc123",
 *   "content": "J'adore la programmation.",
 *   "additional_kwargs": {},
 *   "response_metadata": {
 *     "tokenUsage": {
 *       "completionTokens": 8,
 *       "promptTokens": 19,
 *       "totalTokens": 27
 *     },
 *     "finish_reason": "stop"
 *   },
 *   "tool_calls": [],
 *   "invalid_tool_calls": [],
 *   "usage_metadata": {
 *     "input_tokens": 19,
 *     "output_tokens": 8,
 *     "total_tokens": 27
 *   }
 * }
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Streaming Chunks</strong></summary>
 *
 * ```typescript
 * for await (const chunk of await llm.stream(input)) {
 *   console.log(chunk);
 * }
 * ```
 *
 * ```txt
 * AIMessageChunk {
 *   "content": "J'",
 * }
 * AIMessageChunk {
 *   "content": "adore",
 * }
 * AIMessageChunk {
 *   "content": " la",
 * }
 * AIMessageChunk {
 *   "content": " programmation",
 * }
 * AIMessageChunk {
 *   "content": ".",
 * }
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Aggregate Streamed Chunks</strong></summary>
 *
 * ```typescript
 * import { AIMessageChunk } from '@langchain/core/messages';
 * import { concat } from '@langchain/core/utils/stream';
 *
 * const stream = await llm.stream(input);
 * let full: AIMessageChunk | undefined;
 * for await (const chunk of stream) {
 *   full = !full ? chunk : concat(full, chunk);
 * }
 * console.log(full);
 * ```
 *
 * ```txt
 * AIMessageChunk {
 *   "content": "J'adore la programmation.",
 * }
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Bind tools</strong></summary>
 *
 * ```typescript
 * import { z } from 'zod';
 *
 * const GetWeather = {
 *   name: "GetWeather",
 *   description: "Get the current weather in a given location",
 *   schema: z.object({
 *     location: z.string().describe("The city and state, e.g. San Francisco, CA")
 *   }),
 * }
 *
 * const GetPopulation = {
 *   name: "GetPopulation",
 *   description: "Get the current population in a given location",
 *   schema: z.object({
 *     location: z.string().describe("The city and state, e.g. San Francisco, CA")
 *   }),
 * }
 *
 * const llmWithTools = llm.bindTools([GetWeather, GetPopulation]);
 * const aiMsg = await llmWithTools.invoke(
 *   "Which city is hotter today and which is bigger: LA or NY?"
 * );
 * console.log(aiMsg.tool_calls);
 * ```
 *
 * ```txt
 * [
 *   {
 *     name: 'GetWeather',
 *     args: { location: 'Los Angeles' },
 *     type: 'tool_call',
 *     id: 'call_abc123'
 *   }
 * ]
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Structured Output</strong></summary>
 *
 * ```typescript
 * import { z } from 'zod';
 *
 * const Joke = z.object({
 *   setup: z.string().describe("The setup of the joke"),
 *   punchline: z.string().describe("The punchline to the joke"),
 *   rating: z.number().optional().describe("How funny the joke is, from 1 to 10")
 * }).describe('Joke to tell user.');
 *
 * const structuredLlm = llm.withStructuredOutput(Joke, { name: "Joke" });
 * const jokeResult = await structuredLlm.invoke("Tell me a joke about cats");
 * console.log(jokeResult);
 * ```
 *
 * ```txt
 * {
 *   setup: 'Why did the cat join a band',
 *   punchline: 'Because it wanted to be the purr-cussionist'
 * }
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Usage Metadata</strong></summary>
 *
 * ```typescript
 * const aiMsgForMetadata = await llm.invoke(input);
 * console.log(aiMsgForMetadata.usage_metadata);
 * ```
 *
 * ```txt
 * { input_tokens: 19, output_tokens: 8, total_tokens: 27 }
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Response Metadata</strong></summary>
 *
 * ```typescript
 * const aiMsgForResponseMetadata = await llm.invoke(input);
 * console.log(aiMsgForResponseMetadata.response_metadata);
 * ```
 *
 * ```txt
 * {
 *   tokenUsage: { completionTokens: 8, promptTokens: 19, totalTokens: 27 },
 *   finish_reason: 'stop'
 * }
 * ```
 * </details>
 *
 * <br />
 */
export class ChatAIBadgr extends ChatOpenAICompletions<ChatAIBadgrCallOptions> {
  static lc_name() {
    return "ChatAIBadgr";
  }

  _llmType() {
    return "aibadgr";
  }

  get lc_secrets(): { [key: string]: string } | undefined {
    return {
      aibadgrApiKey: "AIBADGR_API_KEY",
      apiKey: "AIBADGR_API_KEY",
    };
  }

  lc_serializable = true;

  constructor(
    fields?: Partial<
      Omit<OpenAIChatInput, "openAIApiKey" | AIBadgrUnsupportedArgs>
    > &
      BaseChatModelParams & {
        aibadgrApiKey?: string;
        apiKey?: string;
      }
  ) {
    const aibadgrApiKey =
      fields?.apiKey ||
      fields?.aibadgrApiKey ||
      getEnvironmentVariable("AIBADGR_API_KEY");

    if (!aibadgrApiKey) {
      throw new Error(
        `AI Badgr API key not found. Please set the AIBADGR_API_KEY environment variable or provide the key into "aibadgrApiKey"`
      );
    }

    super({
      ...fields,
      model: fields?.model || "gpt-4",
      apiKey: aibadgrApiKey,
      configuration: {
        baseURL: "https://aibadgr.com/api/v1",
      },
    });
  }

  getLsParams(options: this["ParsedCallOptions"]): LangSmithParams {
    const params = super.getLsParams(options);
    params.ls_provider = "aibadgr";
    return params;
  }

  toJSON() {
    const result = super.toJSON();

    if (
      "kwargs" in result &&
      typeof result.kwargs === "object" &&
      result.kwargs != null
    ) {
      delete result.kwargs.openai_api_key;
      delete result.kwargs.configuration;
    }

    return result;
  }

  async completionWithRetry(
    request: OpenAIClient.Chat.ChatCompletionCreateParamsStreaming,
    options?: OpenAICoreRequestOptions
  ): Promise<AsyncIterable<OpenAIClient.Chat.Completions.ChatCompletionChunk>>;

  async completionWithRetry(
    request: OpenAIClient.Chat.ChatCompletionCreateParamsNonStreaming,
    options?: OpenAICoreRequestOptions
  ): Promise<OpenAIClient.Chat.Completions.ChatCompletion>;

  async completionWithRetry(
    request:
      | OpenAIClient.Chat.ChatCompletionCreateParamsStreaming
      | OpenAIClient.Chat.ChatCompletionCreateParamsNonStreaming,
    options?: OpenAICoreRequestOptions
  ): Promise<
    | AsyncIterable<OpenAIClient.Chat.Completions.ChatCompletionChunk>
    | OpenAIClient.Chat.Completions.ChatCompletion
  > {
    delete request.frequency_penalty;
    delete request.presence_penalty;
    delete request.logit_bias;
    delete request.functions;

    return super.completionWithRetry(request, options);
  }
}
