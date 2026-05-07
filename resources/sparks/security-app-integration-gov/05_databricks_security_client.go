package securityclient

import (
    "context"
    "encoding/json"
    "fmt"
    "strings"
    "time"

    "github.com/databricks/databricks-sdk-go"
    "github.com/databricks/databricks-sdk-go/config"
    "github.com/databricks/databricks-sdk-go/service/jobs"
)

type Config struct {
    Host         string
    Token        string
    QueryJobID   int64         // Pre-created Databricks job that runs the query notebook on classic compute
    PollInterval time.Duration // e.g. 5 * time.Second
}

type DatabricksSecurityClient struct {
    w            *databricks.WorkspaceClient
    queryJobID   int64
    pollInterval time.Duration
}

type TriggerResult struct {
    RunID  int64  `json:"run_id"`
    Status string `json:"status"`
}

type TaskState struct {
    Key   string `json:"key"`
    State string `json:"state"`
}

type RunStatus struct {
    State  string      `json:"state"`
    Result string      `json:"result,omitempty"`
    Tasks  []TaskState `json:"tasks,omitempty"`
}

type AlertSummaryRow struct {
    EventType string  `json:"event_type"`
    RiskTier  string  `json:"risk_tier"`
    Count     int64   `json:"count"`
    AvgScore  float64 `json:"avg_score"`
}

type AlertRow struct {
    EventType   string  `json:"event_type"`
    EventID     string  `json:"event_id"`
    Timestamp   string  `json:"timestamp"`
    Principal   string  `json:"principal"`
    RiskScore   float64 `json:"risk_score"`
    RiskTier    string  `json:"risk_tier"`
    RiskReasons string  `json:"risk_reasons"`
    Detail      string  `json:"detail"`
}

type UserProfile struct {
    UserID         string  `json:"user_id"`
    TotalSignins   int64   `json:"total_signins"`
    HighRiskCount  int64   `json:"high_risk_count"`
    AvgRisk        float64 `json:"avg_risk"`
    MaxRisk        float64 `json:"max_risk"`
    NoMfaCount     int64   `json:"no_mfa_count"`
    TotalFailed    int64   `json:"total_failed"`
}

type PipelineStatusRow struct {
    PipelineStage     string  `json:"pipeline_stage"`
    EventType         string  `json:"event_type"`
    RowCount          int64   `json:"row_count"`
    NullRate          float64 `json:"null_rate"`
    Seconds           float64 `json:"seconds"`
    Status            string  `json:"status"`
    RunTimestamp      string  `json:"run_timestamp"`
}

type PipelineAnomalyRow struct {
    PipelineStage   string `json:"pipeline_stage"`
    EventType       string `json:"event_type"`
    RunTimestamp    string `json:"run_timestamp"`
    RowCount        int64  `json:"row_count"`
    NullRate        string `json:"null_rate"`
    AnomalyReasons  string `json:"anomaly_reasons"`
}

func NewDatabricksSecurityClient(cfg Config) (*DatabricksSecurityClient, error) {
    if cfg.QueryJobID == 0 {
        return nil, fmt.Errorf("query_job_id is required")
    }
    if cfg.PollInterval == 0 {
        cfg.PollInterval = 5 * time.Second
    }

    var w *databricks.WorkspaceClient
    var err error

    if cfg.Host != "" && cfg.Token != "" {
        w, err = databricks.NewWorkspaceClient(&databricks.Config{
            Host:  cfg.Host,
            Token: cfg.Token,
        })
    } else {
        w, err = databricks.NewWorkspaceClient(&databricks.Config{
            Profile: config.Profile("DEFAULT"),
        })
    }
    if err != nil {
        return nil, err
    }

    return &DatabricksSecurityClient{
        w:            w,
        queryJobID:   cfg.QueryJobID,
        pollInterval: cfg.PollInterval,
    }, nil
}

func (c *DatabricksSecurityClient) GetAlertSummary(ctx context.Context, database, tableSuffix string) ([]AlertSummaryRow, error) {
    var out []AlertSummaryRow
    err := c.runQueryNotebook(ctx, map[string]string{
        "op":           "get_alert_summary",
        "database":     database,
        "table_suffix": tableSuffix,
    }, &out)
    return out, err
}

func (c *DatabricksSecurityClient) GetTopAlerts(ctx context.Context, database, tableSuffix string, limit int) ([]AlertRow, error) {
    var out []AlertRow
    err := c.runQueryNotebook(ctx, map[string]string{
        "op":           "get_top_alerts",
        "database":     database,
        "table_suffix": tableSuffix,
        "limit":        fmt.Sprintf("%d", limit),
    }, &out)
    return out, err
}

func (c *DatabricksSecurityClient) GetUserProfile(ctx context.Context, database, tableSuffix, userID string) (*UserProfile, error) {
    var out []UserProfile
    err := c.runQueryNotebook(ctx, map[string]string{
        "op":           "get_user_profile",
        "database":     database,
        "table_suffix": tableSuffix,
        "user_id":      userID,
    }, &out)
    if err != nil {
        return nil, err
    }
    if len(out) == 0 {
        return nil, nil
    }
    return &out[0], nil
}

func (c *DatabricksSecurityClient) GetPipelineStatus(ctx context.Context, database, tableSuffix string) ([]PipelineStatusRow, error) {
    var out []PipelineStatusRow
    err := c.runQueryNotebook(ctx, map[string]string{
        "op":           "get_pipeline_status",
        "database":     database,
        "table_suffix": tableSuffix,
    }, &out)
    return out, err
}

func (c *DatabricksSecurityClient) GetPipelineAnomalies(ctx context.Context, database, tableSuffix string) ([]PipelineAnomalyRow, error) {
    var out []PipelineAnomalyRow
    err := c.runQueryNotebook(ctx, map[string]string{
        "op":           "get_pipeline_anomalies",
        "database":     database,
        "table_suffix": tableSuffix,
    }, &out)
    return out, err
}

func (c *DatabricksSecurityClient) TriggerPipeline(ctx context.Context, jobID int64) (*TriggerResult, error) {
    runRef, err := c.w.Jobs.RunNow(ctx, jobs.RunNow{
        JobId: jobID,
    })
    if err != nil {
        return nil, err
    }

    run, err := runRef.Get()
    if err != nil {
        return nil, err
    }

    return &TriggerResult{
        RunID:  run.RunId,
        Status: "triggered",
    }, nil
}

func (c *DatabricksSecurityClient) GetRunStatus(ctx context.Context, runID int64) (*RunStatus, error) {
    run, err := c.w.Jobs.GetRun(ctx, jobs.GetRunRequest{
        RunId: runID,
    })
    if err != nil {
        return nil, err
    }

    status := &RunStatus{
        State: str(run.State.LifeCycleState),
    }
    if run.State != nil && run.State.ResultState != "" {
        status.Result = str(run.State.ResultState)
    }
    for _, t := range run.Tasks {
        state := "unknown"
        if t.State != nil {
            state = str(t.State.LifeCycleState)
        }
        status.Tasks = append(status.Tasks, TaskState{
            Key:   t.TaskKey,
            State: state,
        })
    }
    return status, nil
}

func (c *DatabricksSecurityClient) runQueryNotebook(ctx context.Context, params map[string]string, out any) error {
    runRef, err := c.w.Jobs.RunNow(ctx, jobs.RunNow{
        JobId:          c.queryJobID,
        NotebookParams: params,
    })
    if err != nil {
        return err
    }

    run, err := runRef.Get()
    if err != nil {
        return err
    }

    if err := c.waitForRun(ctx, run.RunId); err != nil {
        return err
    }

    output, err := c.w.Jobs.GetRunOutput(ctx, jobs.GetRunOutputRequest{
        RunId: run.RunId,
    })
    if err != nil {
        return err
    }

    if output.NotebookOutput == nil {
        return fmt.Errorf("run %d returned no notebook output", run.RunId)
    }
    if output.NotebookOutput.Result == "" {
        return fmt.Errorf("run %d returned empty notebook output", run.RunId)
    }

    return json.Unmarshal([]byte(output.NotebookOutput.Result), out)
}

func (c *DatabricksSecurityClient) waitForRun(ctx context.Context, runID int64) error {
    ticker := time.NewTicker(c.pollInterval)
    defer ticker.Stop()

    for {
        run, err := c.w.Jobs.GetRun(ctx, jobs.GetRunRequest{
            RunId: runID,
        })
        if err != nil {
            return err
        }

        state := str(run.State.LifeCycleState)
        switch state {
        case "TERMINATED":
            if run.State != nil && run.State.ResultState != "" && str(run.State.ResultState) != "SUCCESS" {
                return fmt.Errorf("run %d terminated with result=%s", runID, str(run.State.ResultState))
            }
            return nil
        case "SKIPPED", "INTERNAL_ERROR":
            return fmt.Errorf("run %d ended with lifecycle_state=%s", runID, state)
        }

        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-ticker.C:
        }
    }
}

func str[T any](v T) string {
    s := fmt.Sprintf("%v", v)
    return strings.TrimSpace(s)
}
