#include "star_finder.h"

#include "opencv2/imgproc.hpp"
#include <opencv2/calib3d.hpp>


bool build_star_descriptors(const star_detail *stars,
                            u_int32_t count,
                            long width,
                            long height,
                            std::vector<cv::KeyPoint> &keypoints,
                            cv::Mat &descriptors) {
    keypoints.clear();
    descriptors.release();

    if (count < 3) {
        return false;
    }

    const float image_diag = std::sqrt(static_cast<float>(width * width + height * height));
    if (image_diag <= 0.0f) {
        return false;
    }

    std::vector<std::array<float, 3>> rows;
    rows.reserve(count);

    for (u_int32_t i = 0; i < count; i++) {
        const float xi = static_cast<float>(stars[i].x);
        const float yi = static_cast<float>(stars[i].y);

        float best_d2_1 = std::numeric_limits<float>::max();
        float best_d2_2 = std::numeric_limits<float>::max();
        int idx1 = -1;
        int idx2 = -1;

        for (u_int32_t j = 0; j < count; j++) {
            if (j == i) {
                continue;
            }

            const float dx = static_cast<float>(stars[j].x) - xi;
            const float dy = static_cast<float>(stars[j].y) - yi;
            const float d2 = dx * dx + dy * dy;

            if (d2 < best_d2_1) {
                best_d2_2 = best_d2_1;
                idx2 = idx1;
                best_d2_1 = d2;
                idx1 = static_cast<int>(j);
            } else if (d2 < best_d2_2) {
                best_d2_2 = d2;
                idx2 = static_cast<int>(j);
            }
        }

        if (idx1 < 0 || idx2 < 0) {
            continue;
        }

        const float x1 = static_cast<float>(stars[idx1].x);
        const float y1 = static_cast<float>(stars[idx1].y);
        const float x2 = static_cast<float>(stars[idx2].x);
        const float y2 = static_cast<float>(stars[idx2].y);

        const float v1x = x1 - xi;
        const float v1y = y1 - yi;
        const float v2x = x2 - xi;
        const float v2y = y2 - yi;

        const float d1 = std::sqrt(v1x * v1x + v1y * v1y);
        const float d2 = std::sqrt(v2x * v2x + v2y * v2y);
        if (d1 <= 1e-6f || d2 <= 1e-6f) {
            continue;
        }

        float cos_angle = (v1x * v2x + v1y * v2y) / (d1 * d2);
        cos_angle = std::clamp(cos_angle, -1.0f, 1.0f);
        const float angle_norm = std::acos(cos_angle) / static_cast<float>(CV_PI);

        rows.push_back({d1 / image_diag, d2 / image_diag, angle_norm});
        keypoints.emplace_back(cv::Point2f(xi, yi), 5.0f);
    }

    if (rows.size() < 4) {
        keypoints.clear();
        return false;
    }

    descriptors = cv::Mat(static_cast<int>(rows.size()), 3, CV_32F);
    for (int r = 0; r < descriptors.rows; r++) {
        descriptors.at<float>(r, 0) = rows[r][0];
        descriptors.at<float>(r, 1) = rows[r][1];
        descriptors.at<float>(r, 2) = rows[r][2];
    }

    return true;
}

bool build_star_descriptors_generalized(const star_detail *stars,
                                        u_int32_t count,
                                        long width,
                                        long height,
                                        int neighbors,
                                        std::vector<cv::KeyPoint> &keypoints,
                                        cv::Mat &descriptors) {
    keypoints.clear();
    descriptors.release();

    if (neighbors < 2) {
        return false;
    }

    if (count < static_cast<u_int32_t>(neighbors + 1)) {
        return false;
    }

    const float image_diag = std::sqrt(static_cast<float>(width * width + height * height));
    if (image_diag <= 0.0f) {
        return false;
    }

    const int descriptor_dim = 2 * neighbors - 1;
    std::vector<float> all_rows;
    all_rows.reserve(static_cast<size_t>(count) * static_cast<size_t>(descriptor_dim));

    for (u_int32_t i = 0; i < count; i++) {
        const float xi = static_cast<float>(stars[i].x);
        const float yi = static_cast<float>(stars[i].y);

        std::vector<std::pair<float, int>> d2_with_idx;
        d2_with_idx.reserve(count > 0 ? count - 1 : 0);

        for (u_int32_t j = 0; j < count; j++) {
            if (j == i) {
                continue;
            }

            const float dx = static_cast<float>(stars[j].x) - xi;
            const float dy = static_cast<float>(stars[j].y) - yi;
            const float d2 = dx * dx + dy * dy;
            d2_with_idx.emplace_back(d2, static_cast<int>(j));
        }

        if (static_cast<int>(d2_with_idx.size()) < neighbors) {
            continue;
        }

        std::partial_sort(
            d2_with_idx.begin(),
            d2_with_idx.begin() + neighbors,
            d2_with_idx.end(),
            [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
                return a.first < b.first;
            }
        );

        std::vector<cv::Point2f> vectors;
        vectors.reserve(neighbors);

        bool valid = true;
        for (int k = 0; k < neighbors; k++) {
            const int idx = d2_with_idx[k].second;
            const float xk = static_cast<float>(stars[idx].x);
            const float yk = static_cast<float>(stars[idx].y);

            const float vx = xk - xi;
            const float vy = yk - yi;
            const float dist = std::sqrt(vx * vx + vy * vy);
            if (dist <= 1e-6f) {
                valid = false;
                break;
            }

            all_rows.push_back(dist / image_diag);
            vectors.emplace_back(vx, vy);
        }

        if (!valid) {
            for (int rollback = 0; rollback < static_cast<int>(vectors.size()); rollback++) {
                all_rows.pop_back();
            }
            continue;
        }

        for (int k = 0; k < neighbors - 1; k++) {
            const cv::Point2f &v1 = vectors[k];
            const cv::Point2f &v2 = vectors[k + 1];
            const float d1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
            const float d2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);
            float cos_angle = (v1.x * v2.x + v1.y * v2.y) / (d1 * d2);
            cos_angle = std::clamp(cos_angle, -1.0f, 1.0f);
            const float angle_norm = std::acos(cos_angle) / static_cast<float>(CV_PI);
            all_rows.push_back(angle_norm);
        }

        keypoints.emplace_back(cv::Point2f(xi, yi), 5.0f);
    }

    if (keypoints.size() < 4) {
        keypoints.clear();
        return false;
    }

    descriptors = cv::Mat(static_cast<int>(keypoints.size()), descriptor_dim, CV_32F);
    const int rows = descriptors.rows;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < descriptor_dim; c++) {
            descriptors.at<float>(r, c) = all_rows[static_cast<size_t>(r) * static_cast<size_t>(descriptor_dim) + static_cast<size_t>(c)];
        }
    }

    return true;
}


cv::Mat estimate_affine_partial_stars(const std::vector<cv::KeyPoint> &keypoints1, const cv::Mat &descriptors1, 
                                      const std::vector<cv::KeyPoint> &keypoints2, const cv::Mat &descriptors2,
                                      float ratio_threshold, std::vector<cv::DMatch>* inlier_matches) {
    cv::BFMatcher matcher(cv::NORM_L2, false);
    const float ratio_thresh = 0.7f;

    std::vector<std::vector<cv::DMatch>> knn_12;
    std::vector<std::vector<cv::DMatch>> knn_21;
    matcher.knnMatch(descriptors1, descriptors2, knn_12, 2);
    matcher.knnMatch(descriptors2, descriptors1, knn_21, 2);

    std::vector<cv::DMatch> forward_ratio_matches;
    for (const auto &m : knn_12) {
        if (m.size() >= 2 && m[0].distance < ratio_thresh * m[1].distance) {
            forward_ratio_matches.push_back(m[0]);
        }
    }

    std::vector<int> reverse_best_query(descriptors2.rows, -1);
    for (const auto &m : knn_21) {
        if (m.size() >= 2 && m[0].distance < ratio_thresh * m[1].distance) {
            // reverse match: queryIdx is in image2 descriptor set, trainIdx is in image1 descriptor set
            reverse_best_query[m[0].queryIdx] = m[0].trainIdx;
        }
    }

    std::vector<cv::DMatch> mutual_matches;
    mutual_matches.reserve(forward_ratio_matches.size());
    for (const auto &m : forward_ratio_matches) {
        if (m.trainIdx >= 0 && m.trainIdx < static_cast<int>(reverse_best_query.size())
            && reverse_best_query[m.trainIdx] == m.queryIdx) {
            mutual_matches.push_back(m);
        }
    }

    cv::Mat affine_2x3;
    if (mutual_matches.size() >= 3) {
        std::vector<cv::Point2f> points1, points2;
        points1.reserve(mutual_matches.size());
        points2.reserve(mutual_matches.size());

        for (const auto &m : mutual_matches) {
            points1.push_back(keypoints1[m.queryIdx].pt);
            points2.push_back(keypoints2[m.trainIdx].pt);
        }

        cv::Mat inlier_mask;
        affine_2x3 = cv::estimateAffinePartial2D(
            points2,
            points1,
            inlier_mask,
            cv::RANSAC,
            3.0,
            2000,
            0.99,
            10
        );

        if (!affine_2x3.empty() && !inlier_mask.empty() && inlier_matches != nullptr) {
            inlier_matches->reserve(mutual_matches.size());
            for (int i = 0; i < inlier_mask.rows; i++) {
                if (inlier_mask.at<uchar>(i, 0)) {
                    inlier_matches->push_back(mutual_matches[static_cast<size_t>(i)]);
                }
            }
        }
    }

    printf("Built %d descriptors for image1 and %d for image2\n", descriptors1.rows, descriptors2.rows);
    printf("Forward ratio matches: %zu\n", forward_ratio_matches.size());
    printf("Mutual symmetric matches: %zu\n", mutual_matches.size());
    if (inlier_matches != nullptr) {
        printf("RANSAC affine inlier matches: %zu\n", inlier_matches->size());
    }
    return affine_2x3;
}