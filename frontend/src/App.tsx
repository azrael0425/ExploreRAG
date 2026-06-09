import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "sonner";
import { AppShell } from "@/components/layout/AppShell";
import { KnowledgeBasesPage } from "@/pages/KnowledgeBasesPage";
import { WorkspacePage } from "@/pages/WorkspacePage";
import { EvaluationPage } from "@/pages/EvaluationPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<KnowledgeBasesPage />} />
        <Route path="/knowledge-bases/:workspaceId" element={<WorkspacePage />} />
        <Route path="/knowledge-bases/:workspaceId/evaluation" element={<EvaluationPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
      <Toaster
        theme="light"
        position="bottom-right"
        richColors
        toastOptions={{
          duration: 4000,
          className: "text-sm",
        }}
      />
    </QueryClientProvider>
  );
}

export default App;
