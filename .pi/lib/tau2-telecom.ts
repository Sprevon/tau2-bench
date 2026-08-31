import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { existsSync, mkdirSync, realpathSync, writeFileSync } from "node:fs";
import { dirname, delimiter, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { TSchema } from "typebox";

const PROJECT_ROOT = resolve(
	dirname(realpathSync(fileURLToPath(import.meta.url))),
	"../..",
);

const TASK_TOOL_ALLOWLISTS: Readonly<Record<string, readonly string[]>> = {
	"[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]": [
		"get_customer_by_phone",
		"check_status_bar",
		"check_network_status",
		"toggle_roaming",
		"run_speed_test",
	],
};

interface ToolDescriptor {
	name: string;
	description: string;
	parameters: Record<string, unknown>;
	source: "assistant" | "device";
	tool_type: "read" | "write" | "generic" | "think";
	mutates_state: boolean;
}

interface LoadedTask {
	task_id: string;
	ticket: string;
	policy_type: "workflow";
	tool_count: number;
}

interface ToolCallResult {
	content: string;
	error: boolean;
	task_id: string;
	tool_name: string;
}

interface BridgeStatus {
	task_id: string | null;
	loaded: boolean;
	policy_type: "workflow";
	tool_count: number;
}

interface BridgeEnvelope {
	id: number;
	ok: boolean;
	result?: unknown;
	error?: string;
}

export interface TelecomExtensionSnapshot {
	taskId: string | null;
	toolNames: string[];
	activeToolNames: string[];
}

export interface TelecomExtensionOptions {
	taskId?: string;
	autoPrompt?: boolean;
	onTaskLoaded?: (task: LoadedTask, snapshot: TelecomExtensionSnapshot) => void | Promise<void>;
	onToolResult?: (result: ToolCallResult) => void | Promise<void>;
	onEvaluation?: (result: Record<string, unknown>) => void | Promise<void>;
}

export interface TelecomExtensionHandle {
	extension: (pi: ExtensionAPI) => void;
	evaluate: () => Promise<Record<string, unknown>>;
	getSnapshot: () => TelecomExtensionSnapshot;
	dispose: () => void;
}

interface PendingRequest {
	resolve: (value: unknown) => void;
	reject: (reason: Error) => void;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function messageText(message: unknown): string {
	if (!isRecord(message) || message.role !== "assistant") return "";
	if (typeof message.content === "string") return message.content.trim();
	if (!Array.isArray(message.content)) return "";
	return message.content
		.filter((item): item is Record<string, unknown> => isRecord(item) && item.type === "text")
		.map((item) => String(item.text ?? ""))
		.join("\n")
		.trim();
}

function labelFor(name: string): string {
	return name
		.split("_")
		.map((part) => part.charAt(0).toUpperCase() + part.slice(1))
		.join(" ");
}

function selectToolNamesForTask(
	taskId: string,
	descriptors: ToolDescriptor[],
): string[] {
	const allowlist = TASK_TOOL_ALLOWLISTS[taskId];
	if (allowlist === undefined) {
		return descriptors.map((descriptor) => descriptor.name);
	}

	const describedNames = new Set(descriptors.map((descriptor) => descriptor.name));
	const missingNames = allowlist.filter((name) => !describedNames.has(name));
	if (missingNames.length > 0) {
		throw new Error(
			`Task tool allowlist references unavailable tools: ${missingNames.join(", ")}`,
		);
	}
	return [...allowlist];
}

class TelecomBridgeClient {
	private readonly cwd: string;
	private process: ChildProcessWithoutNullStreams | undefined;
	private stdoutBuffer = "";
	private stderrBuffer = "";
	private nextRequestId = 1;
	private pending = new Map<number, PendingRequest>();
	private requestQueue: Promise<void> = Promise.resolve();

	constructor(cwd: string) {
		this.cwd = cwd;
	}

	private start(): ChildProcessWithoutNullStreams {
		if (this.process !== undefined) return this.process;

		const configuredPython = process.env.TAU2_PI_PYTHON?.trim();
		const projectPython = join(this.cwd, ".venv", "bin", "python");
		const python =
			configuredPython || (existsSync(projectPython) ? projectPython : "python");
		const sourcePath = join(this.cwd, "src");
		const pythonPath = process.env.PYTHONPATH;
		const child = spawn(python, ["-m", "tau2.domains.telecom.pi_bridge"], {
			cwd: this.cwd,
			env: {
				...process.env,
				PYTHONPATH: pythonPath ? `${sourcePath}${delimiter}${pythonPath}` : sourcePath,
			},
			stdio: ["pipe", "pipe", "pipe"],
		});
		this.process = child;

		child.stdout.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => this.consumeStdout(chunk));
		child.stderr.setEncoding("utf8");
		child.stderr.on("data", (chunk: string) => {
			this.stderrBuffer = `${this.stderrBuffer}${chunk}`.slice(-8000);
		});
		child.on("error", (error) => this.rejectAll(error));
		child.on("exit", (code, signal) => {
			const diagnostic = this.stderrBuffer.trim();
			const suffix = diagnostic ? `\n${diagnostic}` : "";
			this.rejectAll(
				new Error(
					`tau2 Telecom bridge exited (${code ?? signal ?? "unknown"})${suffix}`,
				),
			);
			this.process = undefined;
		});
		return child;
	}

	private consumeStdout(chunk: string): void {
		this.stdoutBuffer += chunk;
		for (;;) {
			const newline = this.stdoutBuffer.indexOf("\n");
			if (newline < 0) return;
			const line = this.stdoutBuffer.slice(0, newline).trim();
			this.stdoutBuffer = this.stdoutBuffer.slice(newline + 1);
			if (!line) continue;

			let envelope: BridgeEnvelope;
			try {
				envelope = JSON.parse(line) as BridgeEnvelope;
			} catch {
				this.rejectAll(new Error(`Invalid JSON from tau2 Telecom bridge: ${line}`));
				continue;
			}
			const pending = this.pending.get(envelope.id);
			if (pending === undefined) continue;
			this.pending.delete(envelope.id);
			if (envelope.ok) pending.resolve(envelope.result);
			else {
				pending.reject(
					new Error(envelope.error || "Unknown tau2 Telecom bridge error"),
				);
			}
		}
	}

	private rejectAll(error: Error): void {
		for (const pending of this.pending.values()) pending.reject(error);
		this.pending.clear();
	}

	private requestNow<T>(method: string, params: Record<string, unknown> = {}): Promise<T> {
		const child = this.start();
		const id = this.nextRequestId++;
		return new Promise<T>((resolve, reject) => {
			this.pending.set(id, {
				resolve: (value) => resolve(value as T),
				reject,
			});
			child.stdin.write(`${JSON.stringify({ id, method, params })}\n`, (error) => {
				if (error === null || error === undefined) return;
				this.pending.delete(id);
				reject(error);
			});
		});
	}

	request<T>(method: string, params: Record<string, unknown> = {}): Promise<T> {
		const result = this.requestQueue.then(() => this.requestNow<T>(method, params));
		this.requestQueue = result.then(
			() => undefined,
			() => undefined,
		);
		return result;
	}

	stop(): void {
		const child = this.process;
		this.process = undefined;
		if (child !== undefined && !child.killed) child.kill();
		this.rejectAll(new Error("tau2 Telecom bridge stopped"));
	}
}

export function createTau2TelecomExtension(
	options: TelecomExtensionOptions = {},
): TelecomExtensionHandle {
	let client: TelecomBridgeClient | undefined;
	let toolDescriptors: ToolDescriptor[] = [];
	const registeredNames = new Set<string>();
	let loadedTaskId: string | null = null;
	let activeToolNames: string[] = [];

	const getClient = (): TelecomBridgeClient => {
		client ??= new TelecomBridgeClient(PROJECT_ROOT);
		return client;
	};

	const getSnapshot = (): TelecomExtensionSnapshot => ({
		taskId: loadedTaskId,
		toolNames: toolDescriptors.map((descriptor) => descriptor.name),
		activeToolNames: [...activeToolNames],
	});

	const evaluate = async (): Promise<Record<string, unknown>> => {
		if (client === undefined || loadedTaskId === null) {
			throw new Error("No tau2 Telecom task is loaded");
		}
		const result = await client.request<Record<string, unknown>>("evaluate");
		await options.onEvaluation?.(result);
		return result;
	};

	const extension = (pi: ExtensionAPI): void => {
	const registerTelecomTools = async (): Promise<void> => {
		if (toolDescriptors.length === 0) {
			toolDescriptors = await getClient().request<ToolDescriptor[]>("describe_tools");
		}
		for (const descriptor of toolDescriptors) {
			if (registeredNames.has(descriptor.name)) continue;
			registeredNames.add(descriptor.name);
			pi.registerTool({
				name: descriptor.name,
				label: labelFor(descriptor.name),
				description: descriptor.description,
				parameters: descriptor.parameters as TSchema,
				executionMode: "sequential",
				async execute(toolCallId, params, signal) {
					if (signal?.aborted) {
						return {
							content: [{ type: "text", text: "Cancelled" }],
							details: { cancelled: true },
						};
					}
					const argumentsObject = isRecord(params) ? params : {};
					const result = await getClient().request<ToolCallResult>("call_tool", {
						tool_call_id: toolCallId,
						tool_name: descriptor.name,
						arguments: argumentsObject,
					});
					await options.onToolResult?.(result);
					return {
						content: [{ type: "text", text: result.content }],
						details: {
							error: result.error,
							source: descriptor.source,
							toolType: descriptor.tool_type,
							mutatesState: descriptor.mutates_state,
							taskId: result.task_id,
						},
					};
				},
			});
		}
	};

	const loadTelecomTask = async (
		taskId: string,
		ctx: { ui: { notify: (message: string, type?: "info" | "warning" | "error") => void } },
		options: { sendPrompt: boolean },
	): Promise<LoadedTask> => {
		await registerTelecomTools();
		const loaded = await getClient().request<LoadedTask>("load_task", {
			task_id: taskId,
		});
		const availableNames = new Set(pi.getAllTools().map((tool) => tool.name));
		const selectedNames = selectToolNamesForTask(loaded.task_id, toolDescriptors);
			const activeNames = selectedNames.filter((name) => availableNames.has(name));
		if (activeNames.length !== selectedNames.length) {
			const missingNames = selectedNames.filter((name) => !availableNames.has(name));
			throw new Error(
				`Selected Telecom tools are not registered: ${missingNames.join(", ")}`,
			);
		}
			pi.setActiveTools(activeNames);
			activeToolNames = [...activeNames];
			loadedTaskId = loaded.task_id;
		ctx.ui.notify(
			`Activated ${activeNames.length}/${toolDescriptors.length} tau2 Telecom tools`,
			"info",
		);
		pi.setSessionName(`telecom ${loaded.task_id}`);
			if (options.sendPrompt) {
			pi.sendUserMessage(
				`/skill:telecom-solo-support Task ID: ${loaded.task_id}\n` +
					`Policy mode: ${loaded.policy_type}\n\n${loaded.ticket}`,
				{ expandPromptTemplates: true },
			);
			}
			await options.onTaskLoaded?.(loaded, getSnapshot());
			return loaded;
		};

	const writeEvaluation = async (): Promise<void> => {
		const evalOut = process.env.TAU2_TELECOM_EVAL_OUT?.trim();
		if (!evalOut || client === undefined) return;
		mkdirSync(dirname(evalOut), { recursive: true });
		try {
			const result = await evaluate();
			writeFileSync(evalOut, `${JSON.stringify(result, null, 2)}\n`);
		} catch (error) {
			writeFileSync(
				evalOut,
				`${JSON.stringify({ error: String(error), reward: 0.0 }, null, 2)}\n`,
			);
		}
	};

	pi.on("session_start", async (_event, ctx) => {
		try {
			await registerTelecomTools();
			ctx.ui.notify(`Registered ${toolDescriptors.length} tau2 Telecom tools`, "info");
				const taskId = options.taskId?.trim() || process.env.TAU2_TELECOM_TASK_ID?.trim();
				if (taskId) {
					const autoPrompt =
						options.autoPrompt ??
						["1", "true", "yes", "on"].includes(
							(process.env.TAU2_TELECOM_AUTO_PROMPT ?? "").trim().toLowerCase(),
						);
				await loadTelecomTask(taskId, ctx, { sendPrompt: autoPrompt });
			}
		} catch (error) {
			ctx.ui.notify(`Failed to start tau2 Telecom bridge: ${String(error)}`, "error");
		}
	});

	pi.on("session_shutdown", async () => {
		await writeEvaluation();
		client?.stop();
		client = undefined;
		toolDescriptors = [];
		registeredNames.clear();
		loadedTaskId = null;
		activeToolNames = [];
	});

	pi.on("agent_end", async (event) => {
		if (client === undefined || loadedTaskId === null) return;
		const content = messageText(event.messages.at(-1));
		if (!content) return;
		await client.request("record_assistant_text", { content });
	});

	pi.registerCommand("telecom-task", {
		description: "Load a tau2 Telecom solo task: /telecom-task <task-id>",
		handler: async (args, ctx) => {
			const taskId = args.trim();
			if (!taskId) {
				ctx.ui.notify("Usage: /telecom-task <task-id>", "warning");
				return;
			}
			try {
				await loadTelecomTask(taskId, ctx, { sendPrompt: true });
			} catch (error) {
				ctx.ui.notify(`Failed to load Telecom task: ${String(error)}`, "error");
			}
		},
	});

	pi.registerCommand("telecom-status", {
		description: "Show the active tau2 Telecom bridge task",
		handler: async (_args, ctx) => {
			try {
				const status = await getClient().request<BridgeStatus>("status");
				const task = status.task_id ?? "none";
				ctx.ui.notify(
					`Telecom task: ${task}; policy: ${status.policy_type}; tools: ${status.tool_count}`,
					status.loaded ? "info" : "warning",
				);
			} catch (error) {
				ctx.ui.notify(`Failed to query Telecom bridge: ${String(error)}`, "error");
			}
		},
	});
	};

	return {
		extension,
		evaluate,
		getSnapshot,
		dispose: () => {
			client?.stop();
			client = undefined;
			toolDescriptors = [];
			registeredNames.clear();
			loadedTaskId = null;
			activeToolNames = [];
		},
	};
}
