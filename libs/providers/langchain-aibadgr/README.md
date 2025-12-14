# @langchain/aibadgr

This package contains the LangChain.js integrations for AI Badgr (Budget/Utility, OpenAI-compatible).

## Installation

```bash npm2yarn
npm install @langchain/aibadgr @langchain/core
```

## Chat models

This package adds support for AI Badgr's chat model inference.

AI Badgr is a Budget/Utility tier OpenAI-compatible provider that offers cost-effective AI inference.

Set the necessary environment variable (or pass it in via the constructor):

```bash
export AIBADGR_API_KEY=your-api-key
```

Optional: Set a custom base URL (defaults to https://aibadgr.com/api/v1):

```bash
export AIBADGR_BASE_URL=https://aibadgr.com/api/v1
```

```typescript
import { ChatAIBadgr } from "@langchain/aibadgr";
import { HumanMessage } from "@langchain/core/messages";

const model = new ChatAIBadgr({
  apiKey: process.env.AIBADGR_API_KEY, // Default value.
  model: "premium", // or "basic", "normal"
});

const message = new HumanMessage("What is the capital of France?");

const res = await model.invoke([message]);
console.log(res);
```

### Model Names

AI Badgr supports tier-based model naming (recommended):

- **`basic`** - Budget tier for simple tasks
- **`normal`** - Standard tier for most use cases
- **`premium`** - Premium tier for complex tasks (default in examples)

#### Advanced: Power-user model names

For advanced users, specific model names are also supported:

- **`phi-3-mini`** - Maps to basic tier
- **`mistral-7b`** - Maps to normal tier
- **`llama3-8b-instruct`** - Maps to premium tier

OpenAI model names are accepted and mapped automatically to appropriate tiers.

## Development

To develop the `@langchain/aibadgr` package, you'll need to follow these instructions:

### Install dependencies

```bash
pnpm install
```

### Build the package

```bash
pnpm build
```

Or from the repo root:

```bash
pnpm build --filter @langchain/aibadgr
```

### Run tests

Test files should live within a `tests/` file in the `src/` folder. Unit tests should end in `.test.ts` and integration tests should
end in `.int.test.ts`:

```bash
$ pnpm test
$ pnpm test:int
```

### Lint & Format

Run the linter & formatter to ensure your code is up to standard:

```bash
pnpm lint && pnpm format
```

### Adding new entrypoints

If you add a new file to be exported, either import & re-export from `src/index.ts`, or add it to the `exports` field in the `package.json` file and run `pnpm build` to generate the new entrypoint.
