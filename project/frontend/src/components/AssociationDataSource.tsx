import type { LabCatalog, LabResult } from "../lib/lab";

const number = (value:number|undefined) => value===undefined?"Chưa có dữ liệu":value.toLocaleString("vi-VN");

export function AssociationDataSource({catalog}:{catalog:LabCatalog}) {
  const train=catalog.dataset?.splits?.train;
  const fraction=catalog.dataset?.settings?.data?.user_sample_fraction;
  return <section className="lab-section lab-association-data" aria-label="Nguồn dữ liệu Apriori">
    <div className="lab-section-heading"><h2>Dữ liệu dùng để khai phá luật</h2><span className="lab-badge">Toàn bộ Train</span></div>
    <p>Apriori đọc toàn bộ ratings trong <code>train.parquet</code>. <code>movies.parquet</code> cung cấp tên phim; <code>validation.parquet</code> và <code>test.parquet</code> được giữ riêng, không tham gia sinh luật.</p>
    <div className="lab-stat-grid"><div><span>Ratings Train sẵn sàng</span><strong>{number(train?.ratings)}</strong></div><div><span>Users Train sẵn sàng</span><strong>{number(train?.users)}</strong></div><div><span>Lấy mẫu user khi chạy</span><strong>Không</strong></div></div>
    {fraction!==undefined&&fraction<1&&<p className="lab-log-notice">Dùng hết Train đã chuẩn bị. Bản dữ liệu nguồn này được chuẩn bị từ {(fraction*100).toLocaleString("vi-VN")}% người dùng; cần chuẩn bị lại dữ liệu để mở rộng nguồn.</p>}
  </section>;
}

export function AssociationTrainingSummary({result}:{result:LabResult}) {
  const data=result.association_data;
  if(!data)return <p className="lab-muted lab-association-history">Kết quả lịch sử có thể dùng mẫu dữ liệu. Chạy lại Apriori để dùng toàn bộ Train và lưu nhật ký, nguồn dữ liệu của lần chạy mới.</p>;
  const source=result.provenance.association_source;
  const cells:[string,number][]=[
    ["Ratings Train đã quét",data.train_ratings],["Users Train đã dùng",data.train_users],
    ["Lượt thích (rating ≥ 4,0)",data.positive_ratings],["Phim ứng viên",data.candidate_movies],
    ["Lượt thích thuộc phim ứng viên",data.retained_positive_ratings],["Transaction rỗng được giữ",data.empty_transactions],
  ];
  return <section className="lab-section lab-association-data" aria-label="Phạm vi dữ liệu của lần chạy Apriori">
    <div className="lab-section-heading"><h2>Phạm vi khai phá thực tế</h2><span className="lab-badge">Toàn bộ Train · Không lấy mẫu user</span></div>
    <div className="lab-stat-grid">{cells.map(([label,value])=><div key={label}><span>{label}</span><strong>{number(value)}</strong></div>)}</div>
    <p className="lab-muted">Catalog có {number(data.candidate_movies)} phim ứng viên được chọn từ {number(data.positive_movies)} phim có lượt thích trong toàn bộ Train. Support/Confidence/Lift bên dưới là trung bình trên các luật Train đã qua ngưỡng lọc; chưa phải đánh giá chất lượng gợi ý trên dữ liệu mới.</p>
    {source&&<details><summary>Tệp dữ liệu của lần chạy</summary><dl><dt>Ratings dùng khai phá</dt><dd>{source.train_file}</dd><dt>Thông tin phim</dt><dd>{source.movies_file}</dd><dt>Giữ riêng khỏi khai phá</dt>{source.heldout_files.map(file=><dd key={file}>{file}</dd>)}</dl>{source.prepared_user_sample_fraction!==null&&source.prepared_user_sample_fraction<1&&<p className="lab-log-notice">Nguồn đã chuẩn bị từ {(source.prepared_user_sample_fraction*100).toLocaleString("vi-VN")}% người dùng. Lần chạy này dùng hết phần Train của nguồn đó.</p>}</details>}
  </section>;
}
