import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { ChatHistoryResponse } from "@/types";

export function useChatHistory(workspaceId: string) {
  return useQuery({
    queryKey: ["chat-history", workspaceId],
    queryFn: () =>
      api.get<ChatHistoryResponse>(`/rag/chat/${workspaceId}/history`),
    enabled: !!workspaceId,
    staleTime: Infinity, // Don't auto-refetch — we invalidate manually after chat
  });
}

export function useClearChatHistory(workspaceId: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => api.delete(`/rag/chat/${workspaceId}/history`),
    onSuccess: () => {
      queryClient.setQueryData<ChatHistoryResponse>(
        ["chat-history", workspaceId],
        {
          workspace_id: Number(workspaceId),
          messages: [],
          total: 0,
        },
      );
    },
  });
}
