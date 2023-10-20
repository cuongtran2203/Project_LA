> **Hướng dẫn khởi chạy các chương trình**

> Danh sách các chương trình:
>
> classify.py : chương trình này có chức năng infer mô hình dự đoán thể loại phim
>
> train.py: Chương trình này có chức năng huấn luyện mô hình phân loại phim
>
> index.py: chương trình tạo cơ sở dữ liệu cho chương trình search.py
>
> search.py: Chương trình search và đề xuất phim từ mô tả nhập vào

> Cách khởi chạy các chương trình:
>
> Cài đặt một số thư viện cần thiết
>
> ```
> pip install -r requirements.txt
> ```

```
python train.py --training-data tvmaze.sqlite --epochs 5 --batchsize 512
```

```
python index.py --raw-data tvmaze.sqlite
```

```
python classify.py --input-file inputs.txt --output-json-file results.json
```

```
python search.py --input-file inputs.txt --output-json-file results.json
```
