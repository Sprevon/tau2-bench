import {
	createTau2TelecomExtension,
	formatTelecomTaskPrompt,
} from "../lib/tau2-telecom.ts";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

/**
 * Training-only adapter around the canonical Telecom extension.
 *
 * The task execution, tool registration, active-tool allowlist, prompt format,
 * and evaluator remain owned by tau2-telecom. This wrapper only publishes the
 * canonical task prompt and final evaluator result as session entries so the
 * non-interactive Agent-R1 host can await the Pi lifecycle explicitly.
 */
export default function agentR1TrainingExtension(pi: ExtensionAPI): void {
	const taskId = process.env.TAU2_TELECOM_TASK_ID?.trim();
	if (!taskId) {
		throw new Error("TAU2_TELECOM_TASK_ID is required by the Agent-R1 training extension");
	}

	const telecom = createTau2TelecomExtension({
		taskId,
		autoPrompt: false,
		failFastOnStartError: true,
		onTaskLoaded: async (task) => {
			pi.appendEntry("agent-r1-task-prompt", {
				task_id: task.task_id,
				prompt: formatTelecomTaskPrompt(task),
			});
		},
		onEvaluation: async (result) => {
			pi.appendEntry("agent-r1-evaluation", result);
		},
	});

	telecom.extension(pi);
}
