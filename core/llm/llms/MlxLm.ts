import { ChatMessage, CompletionOptions, LLMOptions } from "../../index.js";
import { BaseLLM } from "../index.js";
import { streamSse } from "../stream.js";

interface MlxResponse {
    content?: string;
    error?: string;
    model_id?: string;
    max_tokens?: number;
}

class MlxLm extends BaseLLM {
    static providerName = "mlx_lm";
    static defaultOptions: Partial<LLMOptions> = {
        apiBase: "http://localhost:8000/",
        contextLength: 4096, // Default context length
    };

    constructor(options: LLMOptions) {
        super(options);

        // Query server info to get model details
        this.fetch(new URL("info", this.apiBase))
            .then(async (response) => {
                if (!response.ok) {
                    console.warn(
                        "Error fetching MLX server info:",
                        await response.text()
                    );
                    return;
                }
                const info = await response.json();
                if (info.model_id) {
                    this.model = info.model_id;
                }
                if (info.max_tokens) {
                    this.contextLength = info.max_tokens;
                }
            })
            .catch((err) => {
                console.warn("Failed to fetch MLX server info:", err);
            });
    }

    private _convertArgs(options: CompletionOptions, prompt: string) {
        return {
            prompt,
            max_tokens: options.maxTokens,
            temperature: options.temperature || 0.7,
            top_p: options.topP,
            top_k: options.topK,
            stop: options.stop,
            stream: true,
        };
    }

    protected async *_streamComplete(
        prompt: string,
        signal: AbortSignal,
        options: CompletionOptions,
    ): AsyncGenerator<string> {
        const headers = {
            "Content-Type": "application/json",
            Authorization: `Bearer ${this.apiKey}`,
            ...this.requestOptions?.headers,
        };

        const response = await this.fetch(new URL("generate", this.apiBase), {
            method: "POST",
            headers,
            body: JSON.stringify(this._convertArgs(options, prompt)),
            signal,
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error(`MLX server error: ${error}`);
        }

        try {
            for await (const value of streamSse(response)) {
                const data = value as MlxResponse;
                if (data.error) {
                    throw new Error(`MLX generation error: ${data.error}`);
                }
                if (data.content) {
                    yield data.content;
                }
            }
        } catch (err) {
            if (err instanceof Error && err.name !== "AbortError") {
                console.error("MLX streaming error:", err);
            }
            throw err;
        }
    }

    protected async *_streamChat(
        messages: ChatMessage[],
        signal: AbortSignal,
        options: CompletionOptions,
    ): AsyncGenerator<ChatMessage> {
        // Convert chat messages to a prompt format the model understands
        const prompt = messages
            .map((msg) => `${msg.role}: ${msg.content}`)
            .join("\n");

        for await (const chunk of this._streamComplete(prompt, signal, options)) {
            yield { role: "assistant", content: chunk };
        }
    }

    supportsChat(): boolean {
        return true; // MLX server doesn't natively support chat format
    }

    supportsFim(): boolean {
        return false; // MLX server doesn't support FIM
    }
}

export default MlxLm;
