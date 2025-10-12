
using System;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

[Serializable]
public class PoseFrame {
    public int t;
    public List<List<float>> pose; // [J][C]
}

public class MotionStreamClient : MonoBehaviour
{
    [Header("Server")]
    public string wsUrl = "ws://127.0.0.1:8000/ws/motion";
    public string prompt = "안녕하세요";
    public int fps = 30;

    [Header("Targets (optional IK)")]
    public Transform[] jointTargets; // size J (map a subset to IK targets)

    private ClientWebSocket _ws;
    private CancellationTokenSource _cts;

    async void Start()
    {
        _ws = new ClientWebSocket();
        _cts = new CancellationTokenSource();
        await _ws.ConnectAsync(new Uri(wsUrl), _cts.Token);

        // send request JSON
        string req = "{\"text\":\"" + prompt + "\",\"fps\":"+fps+"}";
        ArraySegment<byte> msg = new ArraySegment<byte>(Encoding.UTF8.GetBytes(req));
        await _ws.SendAsync(msg, WebSocketMessageType.Text, true, _cts.Token);

        _ = ReceiveLoop(); // fire-and-forget
    }

    async Task ReceiveLoop()
    {
        ArraySegment<byte> buf = new ArraySegment<byte>(new byte[1 << 16]);
        StringBuilder sb = new StringBuilder();

        while (_ws.State == WebSocketState.Open)
        {
            WebSocketReceiveResult res = await _ws.ReceiveAsync(buf, _cts.Token);
            if (res.MessageType == WebSocketMessageType.Close) break;

            sb.Append(Encoding.UTF8.GetString(buf.Array, 0, res.Count));
            if (res.EndOfMessage)
            {
                string json = sb.ToString();
                sb.Clear();
                try
                {
                    var frame = JsonUtility.FromJson<PoseFrame>(json);
                    ApplyFrame(frame);
                }
                catch (Exception e)
                {
                    Debug.LogWarning("Parse failed: " + e.Message);
                }
            }
        }
        await _ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "done", _cts.Token);
    }

    void ApplyFrame(PoseFrame f)
    {
        if (jointTargets == null || jointTargets.Length == 0) return;
        int J = Mathf.Min(jointTargets.Length, f.pose.Count);
        for (int j = 0; j < J; j++)
        {
            var tgt = jointTargets[j];
            if (tgt == null) continue;
            var p = f.pose[j];
            if (p.Count >= 3)
            {
                // Simple: interpret as local space positions (meters-ish)
                tgt.localPosition = new Vector3(p[0], p[1], p[2]);
            }
            else if (p.Count == 2)
            {
                tgt.localPosition = new Vector3(p[0], p[1], 0f);
            }
        }
    }

    void OnDestroy()
    {
        try { _cts?.Cancel(); } catch {}
        try { _ws?.Dispose(); } catch {}
    }
}
