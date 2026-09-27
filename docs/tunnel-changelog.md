# Nhật ký đổi cổng / tunnel

Mỗi lần URL public thay đổi (ngrok free đổi link khi restart, đổi cổng, đổi IP), ghi một dòng rồi:

1. Cập nhật `.env` (`PUBLIC_APP_URL`, `PUBLIC_API_URL`, `PUBLIC_AI_URL`)
2. Cập nhật mục **11. Demo online** trong `README.md`
3. Commit + push

| Thời điểm (GMT+7) | Loại | Địa chỉ cũ | Địa chỉ mới | Người ghi |
|---|---|---|---|---|
| 2026-09-24 18:00 | khởi tạo repo | — | local `http://localhost:3000` (FE), `:8000` (BE), `:8001` (AI) | nhóm |
|  |  |  |  |  |

Nếu không có IP tĩnh: cập nhật lại **mỗi sáng thứ Hai** trước khi giảng viên kiểm tra.
